import deepl

auth_key = "3998ee77-3ce9-4b7c-bff0-65db7106880b:fx"
translator = deepl.Translator(auth_key)

result = translator.translate_text("Hello, world!", target_lang="EN")
print(result.text)