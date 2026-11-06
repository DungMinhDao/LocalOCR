-- Remove images and figures: they cost tokens and point at files that aren't exported.
function Image(el) return {} end
function Figure(el) return {} end
