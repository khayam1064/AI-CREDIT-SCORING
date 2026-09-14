with open("dashboard/app.py", "r", encoding="utf-8") as f:
    text = f.read()

# Let's inspect CSS block
css_idx = text.find("<style>")
css_end = text.find("</style>")
if css_idx != -1 and css_end != -1:
    print("Found CSS block length:", css_end - css_idx)
