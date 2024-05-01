# Demo Interface

_______
# Описание полей для пользовательского ввода

|№|Название|Обозначение|Тип|Значения|
|--|--|--|--|--|
|1|Название компании|emp_brand|object|Текстовое поле|
|2|Должность|profession|object|Выпадающий список|
|3|Грейд|grade|object|Выпадающий список|
|4|Город|city|object|Выпадающий список (МСК/СПб/Екб/другое)|
|5|Обязательные требования|mandatory|object|Текстовое поле|
|6|Дополнительные требования|additional|object|Текстовое поле|
|7|Этапы отбора|comp_stages|object|Текстовое поле|
|8|Условия работы|conditions|object|Текстовое поле|
|9|Образование|-|int|group of checkboxes|
|10|Опыт работы|-|int|group of checkboxes|
|11|Формат|-|int|group of checkboxes|
|12|Тестовое задание|test_task|int|checkbox|
|13|Тип занятости|-|int|group of checkboxes|
|14|Оформление|-|int|group of checkboxes|

- фичи 1, 5-8 подаются в LLM (ru-tiny bert) - предсказание - низкая (до 22%)/средняя (22-44%)/высокая (свыше 44%)
- фичи 2-4, 9-14 подаются в Catboost Regressor - предсказание - ожидаемая конверсия в %
_____
# Вторая версия

Только с LLM (Tiny-bert). Анализирует только текст. Запуск [В HuggingFace](https://huggingface.co/spaces/fkonovalenko/llm4career)


