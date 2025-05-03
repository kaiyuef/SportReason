from resiliparse.extract.html2text import extract_plain_text
with open('wikipedia_html/Super_Bowl_XXIII.html','r') as f:
    html = f.read()
print(extract_plain_text(html,list_bullets=False,preserve_formatting=False,main_content=True))