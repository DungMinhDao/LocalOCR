-- Lay out multiple-choice options (A. … B. … C. … D. …) for Word: 4 per line, 2 per line or 1 per line, by the
-- longest option. Rows get the paragraph styles "MCQ 4" / "MCQ 2" / "MCQ 1" and real tabs between options;
-- publish.py adds evenly spaced tab stops to those styles (unless the reference doc already defines them).
local LETTERS = "ABCDEF"
local line_chars = 84  -- text width in characters; set from the page width via -M mcq_line_chars=N

local TAB = pandoc.RawInline("openxml", "<w:r><w:tab/></w:r>")

local function is_break(el)
  return el and (el.t == "Space" or el.t == "SoftBreak" or el.t == "LineBreak")
end

-- "A." / "A)" (or "A.word" glued by OCR) → the text after the marker, else nil
local function marker(el, k)
  if not el or el.t ~= "Str" then return nil end
  return el.text:match("^" .. LETTERS:sub(k, k) .. "[.)](.*)$")
end

local function trim(inl)
  while #inl > 0 and is_break(inl[1]) do inl:remove(1) end
  while #inl > 0 and is_break(inl[#inl]) do inl:remove(#inl) end
  return inl:walk({ SoftBreak = function() return pandoc.Space() end, LineBreak = function() return pandoc.Space() end })
end

local function slice(inl, i, j)
  local out = pandoc.Inlines({})
  for x = i, j do out:insert(inl[x]) end
  return out
end

local function row(cols, inl)
  return pandoc.Div({ pandoc.Para(inl) }, { ["custom-style"] = "MCQ " .. cols })
end

local function layout(para)
  local inl = para.content
  local starts, k = {}, 1
  for i = 1, #inl do
    if k <= #LETTERS and (i == 1 or is_break(inl[i - 1])) then
      local rest = marker(inl[i], k)
      if rest then starts[k] = { i = i, rest = rest }; k = k + 1 end
    end
  end
  if #starts < 3 then return nil end  -- "A. Lincoln said…" is not a question

  local opts, longest = {}, 0
  for j, s in ipairs(starts) do
    local stop = starts[j + 1] and starts[j + 1].i - 1 or #inl
    local body = slice(inl, s.i + 1, stop)
    if s.rest ~= "" then body:insert(1, pandoc.Str(s.rest)) end
    body = trim(body)
    local label = pandoc.Inlines({ pandoc.Str(LETTERS:sub(j, j) .. "."), pandoc.Space() })
    label:extend(body)
    opts[j] = label
    longest = math.max(longest, #pandoc.utils.stringify(label))
  end

  local out = {}
  local prefix = slice(inl, 1, starts[1].i - 1)
  local own_line = starts[1].i > 1 and (inl[starts[1].i - 1].t ~= "Space" or #pandoc.utils.stringify(prefix) > 12)
  prefix = trim(prefix)
  if own_line then
    table.insert(out, pandoc.Para(prefix))
  elseif #prefix > 0 then
    local first = pandoc.Inlines(prefix)
    first:insert(pandoc.Space())
    first:extend(opts[1])
    opts[1] = first
  end

  local first_len = #pandoc.utils.stringify(opts[1])
  local cols = 1
  if math.max(longest, first_len) + 3 <= line_chars / 4 then
    cols = 4
  elseif math.max(longest, first_len) + 3 <= line_chars / 2 then
    cols = 2
  end
  for j = 1, #opts, cols do
    local r = pandoc.Inlines({})
    for x = j, math.min(j + cols - 1, #opts) do
      if x > j then r:insert(TAB) end
      r:extend(opts[x])
    end
    table.insert(out, row(cols, r))
  end
  return out
end

local function is_text(b)
  return b and (b.t == "Para" or b.t == "Plain")
end

local function Blocks(blocks)
  -- Options on separate lines with blank lines between them arrive as separate paragraphs: rejoin them.
  local merged = pandoc.Blocks({})
  for _, b in ipairs(blocks) do
    local prev = merged[#merged]
    local k = is_text(b) and LETTERS:find(((b.content[1] or {}).text or ""):match("^([B-F])[.)]") or "", 1, true)
    if k and k > 1 and is_text(prev) and pandoc.utils.stringify(prev):find("%f[%w]" .. LETTERS:sub(k - 1, k - 1) .. "[.)]") then
      prev.content:insert(pandoc.SoftBreak())
      prev.content:extend(b.content)
    else
      merged:insert(b)
    end
  end
  local out = pandoc.Blocks({})
  for _, b in ipairs(merged) do
    local laid = is_text(b) and layout(b)
    if laid then out:extend(laid) else out:insert(b) end
  end
  return out
end

-- "1. A. … B. …" is read as a numbered list; the styled rows would lose the number, so write it out as text.
local function OrderedList(list)
  local first = list.content[1] and list.content[1][1]
  if not (is_text(first) and marker(first.content[1], 1)) then return nil end
  local out, n = pandoc.Blocks({}), list.start
  local delim = list.delimiter == "OneParen" and ")" or "."
  for _, item in ipairs(list.content) do
    local head = item[1]
    if is_text(head) then
      head = pandoc.Para({ pandoc.Str(n .. delim), pandoc.Space(), table.unpack(head.content) })
      out:insert(head)
      for x = 2, #item do out:insert(item[x]) end
    else
      out:extend(item)
    end
    n = n + 1
  end
  return out
end

return {
  { OrderedList = OrderedList },
  { Meta = function(m) if m.mcq_line_chars then line_chars = tonumber(pandoc.utils.stringify(m.mcq_line_chars)) end end },
  { Blocks = Blocks },
}
