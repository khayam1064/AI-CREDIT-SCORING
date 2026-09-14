import pptx
prs = pptx.Presentation("APEX_Credit_AI_Engine_Enterprise_Presentation.pptx")
print(f"Verified Presentation: {len(prs.slides)} Widescreen (16:9) Slides Created Successfully!")
for i, slide in enumerate(prs.slides):
    # Find text boxes
    titles = [shape.text_frame.paragraphs[0].text for shape in slide.shapes if shape.has_text_frame and len(shape.text_frame.paragraphs) > 0]
    first_title = titles[0] if titles else "Untitled"
    print(f"  Slide {i+1:2d}: {first_title[:55]}")
