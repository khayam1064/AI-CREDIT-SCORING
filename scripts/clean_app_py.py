with open('dashboard/app.py', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

# Fix header and title
text = text.replace('Pak-Credit AI  Digital Lending Platform', 'APEX CREDIT OS · Enterprise Risk & Digital Lending Platform')
text = text.replace('Pak-Credit AI · Digital Lending Platform', 'APEX CREDIT OS · Enterprise Risk & Digital Lending Platform')
text = text.replace('page_icon="dY?"', 'page_icon="🏦"')
text = text.replace('page_icon="dY?"', 'page_icon="🏦"')
text = text.replace('1. dY"', '1. 📱')
text = text.replace('2. dY\'', '2. 💼')
text = text.replace('1. dY"', '1. 📱')
text = text.replace('2. dY\'', '2. 💼')
text = text.replace('dY?', '🏦')
text = text.replace('dY? ', '🏦 ')
text = text.replace('dY?', '🏦')
text = text.replace('dY"', '📱')
text = text.replace('dY\'', '💼')
text = text.replace('\ufffd', '')

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Successfully cleaned app.py!')
