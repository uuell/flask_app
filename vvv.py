# from googletrans import Translator

# translator = Translator()
# async def translate_text(text, src, dest):
#     try:
#         translated = await translator.translate(text, src=src, dest=dest)
#         print(translated.text)
#     except Exception as e:
#         print(f"Error during translation: {e}")
#         return text
    
# translate_text("안녕하세요", 'ko', 'en')

import asyncio
from googletrans import Translator

async def main():
    translator = Translator()
    try:
        translated = await translator.translate("지금도 장그러운데 이결 한 시간이나 만져야 한다고", src='ko', dest='en')
        print(translated.text)
    except Exception as e:
        print(f"Error during translation: {e}")
        return

asyncio.run(main())
