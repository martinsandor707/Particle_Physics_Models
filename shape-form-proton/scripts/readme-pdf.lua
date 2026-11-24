-- Permit long feature names, paths and formulas to wrap in PDF table cells.
-- This changes LaTeX rendering only; README.md remains the source of truth.
local escapes = {
  ['\\'] = '\\textbackslash{}', ['{'] = '\\{', ['}'] = '\\}',
  ['_'] = '\\_', ['%'] = '\\%', ['&'] = '\\&', ['#'] = '\\#',
  ['$'] = '\\$', ['^'] = '\\textasciicircum{}', ['~'] = '\\textasciitilde{}'
}

local break_after = {
  ['_'] = true, ['/'] = true, ['.'] = true, [','] = true,
  ['+'] = true, ['*'] = true, ['('] = true, [')'] = true,
  ['='] = true, [' '] = true, ['-'] = true, ['−'] = true
}

function Code(el)
  if not FORMAT:match('latex') then return nil end
  local parts = {}
  for char in el.text:gmatch('[%z\1-\127\194-\244][\128-\191]*') do
    table.insert(parts, escapes[char] or char)
    if break_after[char] then
      table.insert(parts, '\\allowbreak{}')
    end
  end
  return pandoc.RawInline('latex', '\\texttt{' .. table.concat(parts) .. '}')
end
