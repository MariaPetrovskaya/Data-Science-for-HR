import gradio as gr
import os
import shutil

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GlobalState:
    """
    Class to store global variables
    """
    result_file_path = os.path.join(os.path.dirname(__file__), 'download/')


def prepare_data():
    pass


def run_bert():
    pass


def predict():
    pass


def save():
    pass


def main():
    """
    TODO: Fix progress bars
    """
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'download/'), ignore_errors=True)
    os.mkdir(os.path.join(os.path.dirname(__file__), 'download/'))
    with gr.Blocks() as demo:
        with gr.Tab("Load"):
            with gr.Row():
                gr.Markdown(
                    """
                    # input position info, write a text description
                    # Then press **Run!**
                    # Have fun:)
                    """)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        brand = gr.Textbox(label='Название компании', value='Yandex')
                    with gr.Row():
                        position = gr.Dropdown(label='Название должности', choices=['data scientist', 'HR'], value='HR')
                    with gr.Row():
                        grade = gr.Dropdown(label='Грейд', choices=['senior', 'lead'], value='senior')
                    with gr.Row():
                        city = gr.Dropdown(label='Город',
                                           choices=['Москва', 'Санкт-Петербург', 'Екатеринбург', 'Другой'],
                                           value='Магадан')
                    with gr.Row():
                        test_task = gr.Checkbox(label="Тестовое задание")
                    with gr.Row():
                        edu = gr.CheckboxGroup(choices=['Релевантное', 'Около-релевантное', 'Другое'],
                                               label='Образование')
                    with gr.Row():
                        exp = gr.CheckboxGroup(choices=['Релевантный', 'Около-релевантный', 'Учебный'],
                                               label='Опыт работы')

                with gr.Column():
                    with gr.Row():
                        place = gr.CheckboxGroup(choices=['Офис', 'Удаленка', 'Гибрид'],
                                                 label='Формат')
                    with gr.Row():
                        empl = gr.CheckboxGroup(choices=['Волонтерство', 'Стажировка', 'Полная',
                                                         'Частичная', 'Проектная'],
                                                label='Тип занятости')
                    with gr.Row():
                        contr = gr.CheckboxGroup(choices=['Трудовой договор', 'ГПХ', 'ИП',
                                                          'Контракт', 'Самозанятость', 'Кабала'],
                                                 label='Оформление')
                    with gr.Row():
                        requ = gr.Textbox(label='Обязательные требования')
                    with gr.Row():
                        add = gr.Textbox(label='Дополнительные требования')
                    with gr.Row():
                        comp = gr.Textbox(label='Этапы отбора')
                    with gr.Row():
                        comp = gr.Textbox(label='Условия работы', value='Платный туалет и битье батогами по пятницам')

                with gr.Column():
                    with gr.Row():
                        process_button = gr.Button("Run!")
                    with gr.Row():
                        output_1 = gr.Textbox(label='CatBoost Result')
                    with gr.Row():
                        output_2 = gr.Textbox(label='LLM Result')

    demo.launch(share=True, allowed_paths=[GlobalState.result_file_path])


if __name__ == "__main__":
    main()
