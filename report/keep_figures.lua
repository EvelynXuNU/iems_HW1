-- Keep each image and its manually numbered caption on the same page.
function Pandoc(doc)
  local blocks = pandoc.List()
  local i = 1
  while i <= #doc.blocks do
    local b = doc.blocks[i]
    local next = doc.blocks[i + 1]
    if b.t == "Para" and #b.content == 1 and b.content[1].t == "Image"
      and next and next.t == "Para"
      and pandoc.utils.stringify(next):match("^Figure ") then
      blocks:insert(pandoc.RawBlock("latex", "\\noindent\\begin{minipage}{\\linewidth}"))
      blocks:insert(b)
      blocks:insert(next)
      blocks:insert(pandoc.RawBlock("latex", "\\end{minipage}\\par\\medskip"))
      i = i + 2
    else
      blocks:insert(b)
      i = i + 1
    end
  end
  doc.blocks = blocks
  return doc
end
