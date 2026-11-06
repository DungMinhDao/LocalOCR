// Apple Vision OCR helper for LocalOCR.
//
// Usage: vision-ocr [--mode document|text] [--fast] [--boxes] [--langs en-US,vi-VT] img1.png img2.png ...
// Prints one JSON object per image on stdout:
//   {"path": ..., "markdown": ..., "text": ..., "seconds": ..., "mode": ...}
// --boxes (text mode) adds "lines": [{"t", "x", "y", "w", "h"}] as fractions of the image, top-left origin.
//
// "document" mode uses RecognizeDocumentsRequest (macOS 26+) which understands
// paragraphs, lists and tables. "text" mode uses the classic VNRecognizeTextRequest.

import Foundation
import Vision
import CoreGraphics
import ImageIO

struct Item {
    var top: Double      // distance from top of page, 0...1
    var left: Double
    var markdown: String
}

func emit(_ dict: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: dict, options: []),
       let s = String(data: data, encoding: .utf8) {
        print(s)
        fflush(stdout)
    }
}

func loadCGImage(_ url: URL) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

func escapeCell(_ s: String) -> String {
    s.replacingOccurrences(of: "|", with: "\\|").replacingOccurrences(of: "\n", with: " ")
}

// MARK: - Classic text recognition

func recognizeText(url: URL, fast: Bool, langs: [String]) throws -> (String, String, [[String: Any]]) {
    guard let image = loadCGImage(url) else {
        throw NSError(domain: "vision-ocr", code: 1, userInfo: [NSLocalizedDescriptionKey: "cannot read image"])
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = fast ? .fast : .accurate
    request.usesLanguageCorrection = !fast
    request.automaticallyDetectsLanguage = langs.isEmpty
    if !langs.isEmpty { request.recognitionLanguages = langs }
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    try handler.perform([request])
    let observations = request.results ?? []

    var lines: [(box: CGRect, text: String)] = []
    for o in observations {
        if let c = o.topCandidates(1).first { lines.append((o.boundingBox, c.string)) }
    }
    let text = lines.map { $0.text }.joined(separator: "\n")
    let boxes: [[String: Any]] = lines.map { l in
        ["t": l.text, "x": l.box.minX, "y": 1 - l.box.maxY, "w": l.box.width, "h": l.box.height]
    }

    // Group lines into paragraphs using vertical gaps.
    var paragraphs: [String] = []
    var current: [String] = []
    var prev: CGRect? = nil
    for l in lines {
        if let p = prev {
            let gap = p.minY - l.box.maxY
            let h = max(p.height, l.box.height)
            if gap > h * 0.6 || l.box.maxY > p.maxY + h {   // big gap or jumped to a new column
                paragraphs.append(current.joined(separator: "  \n"))
                current = []
            }
        }
        current.append(l.text)
        prev = l.box
    }
    if !current.isEmpty { paragraphs.append(current.joined(separator: "  \n")) }
    return (paragraphs.joined(separator: "\n\n"), text, boxes)
}

// MARK: - Structured document recognition (macOS 26+)

@available(macOS 26.0, *)
func recognizeDocument(url: URL) async throws -> (String, String) {
    let request = RecognizeDocumentsRequest()
    let observations = try await request.perform(on: url)
    guard let doc = observations.first?.document else { return ("", "") }

    var items: [Item] = []
    var blocked: [CGRect] = []   // regions already covered by tables / lists

    for table in doc.tables {
        let r = table.boundingRegion.boundingBox.cgRect
        blocked.append(r)
        var md: [String] = []
        for (i, row) in table.rows.enumerated() {
            let cells = row.map { escapeCell($0.content.text.transcript) }
            md.append("| " + cells.joined(separator: " | ") + " |")
            if i == 0 { md.append("|" + String(repeating: " --- |", count: max(cells.count, 1))) }
        }
        items.append(Item(top: 1 - r.maxY, left: r.minX, markdown: md.joined(separator: "\n")))
    }
    for list in doc.lists {
        let r = list.boundingRegion.boundingBox.cgRect
        blocked.append(r)
        // item.content already includes the marker ("1.", "-", "B."); itemString is the bare text.
        let md = list.items.map { item -> String in
            let body = item.content.text.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            let bare = item.itemString.trimmingCharacters(in: .whitespacesAndNewlines)
            let marker = body.hasSuffix(bare) ? String(body.dropLast(bare.count)).trimmingCharacters(in: .whitespaces) : ""
            if marker.isEmpty || "•◦▪■●-–*".contains(marker) { return "- " + bare }
            return marker + " " + bare
        }.joined(separator: "\n")
        items.append(Item(top: 1 - r.maxY, left: r.minX, markdown: md))
    }
    let title = doc.title?.transcript
    for p in doc.paragraphs {
        let r = p.boundingRegion.boundingBox.cgRect
        let c = CGPoint(x: r.midX, y: r.midY)
        if blocked.contains(where: { $0.insetBy(dx: -0.005, dy: -0.005).contains(c) }) { continue }
        var md = p.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        if md.isEmpty { continue }
        if let t = title, t == md { md = "# " + md }
        items.append(Item(top: 1 - r.maxY, left: r.minX, markdown: md))
    }
    // Reading order: top to bottom, then left to right for items on the same band.
    items.sort { a, b in
        if abs(a.top - b.top) < 0.01 { return a.left < b.left }
        return a.top < b.top
    }
    let markdown = items.map { $0.markdown }.joined(separator: "\n\n")
    return (markdown, doc.text.transcript)
}

// MARK: - main

@main
struct VisionOCR {
    static func main() async {
        var args = Array(CommandLine.arguments.dropFirst())
        var mode = "document"
        var fast = false
        var boxes = false
        var langs: [String] = []
        var paths: [String] = []
        while !args.isEmpty {
            let a = args.removeFirst()
            switch a {
            case "--mode": mode = args.isEmpty ? mode : args.removeFirst()
            case "--fast": fast = true
            case "--boxes": boxes = true
            case "--langs":
                let v = args.isEmpty ? "" : args.removeFirst()
                langs = v.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            default: paths.append(a)
            }
        }
        if #unavailable(macOS 26.0) { if mode == "document" { mode = "text" } }

        for path in paths {
            let url = URL(fileURLWithPath: path)
            let t0 = Date()
            do {
                var md = "", text = "", lines: [[String: Any]] = []
                if mode == "document", #available(macOS 26.0, *) {
                    (md, text) = try await recognizeDocument(url: url)
                } else {
                    (md, text, lines) = try recognizeText(url: url, fast: fast, langs: langs)
                }
                var out: [String: Any] = ["path": path, "markdown": md, "text": text, "mode": mode,
                                          "seconds": Date().timeIntervalSince(t0)]
                if boxes { out["lines"] = lines }
                emit(out)
            } catch {
                emit(["path": path, "error": error.localizedDescription, "mode": mode,
                      "seconds": Date().timeIntervalSince(t0)])
            }
        }
    }
}
