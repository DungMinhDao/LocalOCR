-- Parse raw HTML tables (VLM output) into real pandoc tables: DOCX/PPTX/ODT writers drop raw HTML.
-- Requires the reader to run without markdown_in_html_blocks, so a whole <table> arrives as one RawBlock.
-- Cell text that holds Markdown or TeX ($x$, \(..\), **b**, [u]{.underline}) is re-read as Markdown.
local MD = "markdown+tex_math_dollars+tex_math_single_backslash"

local function has_markup(s)
  return s:find("\\(", 1, true) or s:find("$", 1, true) or s:find("**", 1, true) or s:find("]{", 1, true)
end

local function remark(block)
  local s = pandoc.utils.stringify(block)
  if not has_markup(s) then return nil end
  s = s:gsub("^(%w+)([.)])", "%1\\%2"):gsub("^([-+*#]%s)", "\\%1"):gsub("^>", "\\>")  -- "1. text" is a cell, not a list
  local doc = pandoc.read(s, MD)
  if #doc.blocks == 1 and (doc.blocks[1].t == "Para" or doc.blocks[1].t == "Plain") then
    return pandoc.Plain(doc.blocks[1].content)
  end
end

function RawBlock(el)
  if el.format ~= "html" or not el.text:find("<table", 1, true) then return nil end
  local blocks = pandoc.read(el.text, "html").blocks
  return pandoc.Div(blocks):walk({ Plain = remark, Para = remark }).content
end
