import gradio as gr
import os
import shutil
import json
from ml import VacancyAnalyzer


class GlobalState:
    """
    Class to store global variables
    """
    result_file_path = os.path.join(os.path.dirname(__file__), 'result/archive.json')
    result_dir = os.path.join(os.path.dirname(__file__), 'result')
    bert_path = os.path.join(os.path.dirname(__file__), 'tiny.pt')
    catboost_path = os.path.join(os.path.dirname(__file__), 'best_cat.joblib')
    conv_classes = {0: 'low',
                    1: 'middle',
                    2: 'high'
                    }
    default_data = {'id': 'a0000',
            'emp_brand': 'Рога и Копыта',
            'test_task': 0,
            'grade': 'junior',
            'profession': 'developer',
            'location': 'Moscow',
            'relevant_exp': 0,
            'near_exp': 0,
            'student_exp': 0,
            'relevant_edu': 0,
            'near_edu': 0,
            'other_edu': 0,
            'remote': 0,
            'office': 0,
            'hybrid': 0,
            'ind_enterp': 0,
            'temporary': 0,
            'gph': 0,
            'permanent': 0,
            'self_empl': 0,
            'internship': 0,
            'project': 0,
            'volunteering': 0,
            'part_time': 0,
            'full_time': 0,
            'mandatory': 'Знание языка Python для микроконтроллеров. Не менее 20 лет опыта работы с GPT4',
            'additional': 'Не брезгливый. Не бояться костей животных',
            'comp_stages': 'Собеседование. 15 тестовых заданий. Победить Лернейскую Гидру',
            'work_conditions': 'Крепостное право. Платный туалет во дворе офиса и битье батогами по пятницам.',
            'conversion': 0,
            'conversion_class': 'unknown'
            }
    data = None


def cid(txt):
    GlobalState.data['id'] = txt


def cbrand(txt):
    GlobalState.data['emp_brand'] = txt


def cmand(txt):
    GlobalState.data['mandatory'] = txt


def cadd(txt):
    GlobalState.data['additional'] = txt


def ccomp(txt):
    GlobalState.data['comp_stages'] = txt


def ccond(txt):
    GlobalState.data['work_conditions'] = txt


def cposition(dd):
    GlobalState.data['profession'] = dd


def cgrade(dd):
    GlobalState.data['grade'] = dd


def ccity(dd):
    GlobalState.data['location'] = dd


def ctask(chck):
    GlobalState.data['test_task'] = int(chck)


def cedu(chck):
    if 'Relevant' in chck:
        GlobalState.data['relevant_edu'] = 1
    else:
        GlobalState.data['relevant_edu'] = 0
    if 'Near-relevant' in chck:
        GlobalState.data['near_edu'] = 1
    else:
        GlobalState.data['near_edu'] = 0
    if 'Other' in chck:
        GlobalState.data['other_edu'] = 1
    else:
        GlobalState.data['other_edu'] = 0


def cexp(chck):
    if 'Relevant' in chck:
        GlobalState.data['relevant_exp'] = 1
    else:
        GlobalState.data['relevant_exp'] = 0
    if 'Near-relevant' in chck:
        GlobalState.data['near_exp'] = 1
    else:
        GlobalState.data['near_exp'] = 0
    if 'Student' in chck:
        GlobalState.data['student_exp'] = 1
    else:
        GlobalState.data['student_exp'] = 0


def cplace(chck):
    if 'Office' in chck:
        GlobalState.data['office'] = 1
    else:
        GlobalState.data['office'] = 0
    if 'Remote' in chck:
        GlobalState.data['remote'] = 1
    else:
        GlobalState.data['remote'] = 0
    if 'Hybrid' in chck:
        GlobalState.data['hybrid'] = 1
    else:
        GlobalState.data['hybrid'] = 0


def cempl(chck):
    if 'Volunteering' in chck:
        GlobalState.data['volunteering'] = 1
    else:
        GlobalState.data['volunteering'] = 0
    if 'Internship' in chck:
        GlobalState.data['internship'] = 1
    else:
        GlobalState.data['internship'] = 0
    if 'Full-time' in chck:
        GlobalState.data['full_time'] = 1
    else:
        GlobalState.data['full_time'] = 0
    if 'Part-time' in chck:
        GlobalState.data['part_time'] = 1
    else:
        GlobalState.data['part_time'] = 0
    if 'Project' in chck:
        GlobalState.data['project'] = 1
    else:
        GlobalState.data['project'] = 0


def ccontr(chck):
    if 'Contract' in chck:
        GlobalState.data['permanent'] = 1
    else:
        GlobalState.data['permanent'] = 0
    if 'Civil contract' in chck:
        GlobalState.data['gph'] = 1
    else:
        GlobalState.data['gph'] = 0
    if 'Individual Entrepreneur' in chck:
        GlobalState.data['ind_enterp'] = 1
    else:
        GlobalState.data['ind_enterp'] = 0
    if 'Temporary contract' in chck:
        GlobalState.data['temporary'] = 1
    else:
        GlobalState.data['temporary'] = 0
    if 'Self-Employed' in chck:
        GlobalState.data['self_empl'] = 1
    else:
        GlobalState.data['self_empl'] = 0


def submit(chk):
    # print(GlobalState.data)
    return gr.update("Run!", visible=True)


def append_to_json(_dict, path):
    with open(path, 'ab+') as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(json.dumps([_dict]).encode())
        else:
            f.seek(-1, 2)
            f.truncate()
            f.write(' , '.encode())
            f.write(json.dumps(_dict).encode())
            f.write(']'.encode())


def predict(btn):
    analyzer = VacancyAnalyzer(GlobalState.bert_path, GlobalState.catboost_path, GlobalState.data)
    c_coeff = analyzer.predict()
    gr.Info('Conversion coefficient calculation finished')
    c_class = GlobalState.conv_classes[int(analyzer.classify()[0])]
    gr.Info('Text analyzing finished')
    out_1 = f'Predicted by vacancy parameters conversion - {c_coeff:.0%}'
    out_2 = f'Predicted by vacancy description conversion - {c_class}'
    GlobalState.data['conversion'] = c_coeff
    GlobalState.data['conversion_class'] = c_class
    fid = GlobalState.result_file_path
    append_to_json(GlobalState.data, fid)
    GlobalState.data = GlobalState.default_data
    link = GlobalState.result_file_path
    return gr.update(value=out_1), gr.update(value=out_2), gr.update(link="/file=" + link, visible=True)


def save(btn):
    link = GlobalState.result_file_path
    return gr.update(link="/file=" + link)


def main():
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'result/'), ignore_errors=True)
    os.mkdir(os.path.join(os.path.dirname(__file__), 'result/'))
    GlobalState.data = GlobalState.default_data
    with gr.Blocks() as demo:
        with gr.Tab("Load"):
            with gr.Row():
                gr.Markdown(
                    """
                    # input position info, write a text description
                    # Then press **Run!**
                    # Have fun 👾
                    """)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            brand = gr.Textbox(label='Company name', value=None)
                        with gr.Column():
                            vid = gr.Textbox(label='Vacancy ID', value=None)
                    with gr.Row():
                        position = gr.Dropdown(label='Profession', choices=['developer',
                                                                            'devops',
                                                                            'ds', 'analyst',
                                                                            'manager',
                                                                            'tester', 'designer',
                                                                            'marketolog', 'support',
                                                                            'hr'], value=None)
                    with gr.Row():
                        grade = gr.Dropdown(label='Grade', choices=['intern',
                                                                    'junior',
                                                                    'middle', 'senior', 'lead'], value=None)
                    with gr.Row():
                        city = gr.Dropdown(label='City',
                                           choices=['Moscow', 'Saint-Petersburg', 'Yekaterinburg', 'Other'],
                                           value=None)
                    with gr.Row():
                        test_task = gr.Checkbox(label="Test Task", interactive=True)
                    with gr.Row():
                        edu = gr.CheckboxGroup(choices=['Relevant', 'Near-relevant', 'Other'],
                                               label='Education', interactive=True)
                    with gr.Row():
                        exp = gr.CheckboxGroup(choices=['Relevant', 'Near-relevant', 'Student'],
                                               label='Experience', interactive=True)

                with gr.Column():
                    with gr.Row():
                        place = gr.CheckboxGroup(choices=['Office', 'Remote', 'Hybrid'],
                                                 label='Work Format', interactive=True)
                    with gr.Row():
                        empl = gr.CheckboxGroup(choices=['Volunteering', 'Internship', 'Full-time',
                                                         'Part-time', 'Project'],
                                                label='Employment', interactive=True)
                    with gr.Row():
                        contr = gr.CheckboxGroup(choices=['Contract', 'Civil contract', 'Individual Entrepreneur',
                                                          'Temporary contract', 'Self-Employed'],
                                                 label='Contract Type', interactive=True)
                    with gr.Row():
                        req = gr.Textbox(label='Mandatory')
                    with gr.Row():
                        add = gr.Textbox(label='Addition')
                    with gr.Row():
                        comp = gr.Textbox(label='Competition stage')
                    with gr.Row():
                        cond = gr.Textbox(label='Work conditions')

                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            ready = gr.Checkbox(label='Data Filled')
                        with gr.Column():
                            process_button = gr.Button("Run!", visible=False, interactive=True)
                    with gr.Row():
                        output_1 = gr.Textbox(label='CatBoost Result')
                    with gr.Row():
                        output_2 = gr.Textbox(label='LLM Result')
                    with gr.Row():
                        download_button = gr.Button("JSON Archive", visible=False)

        vid.change(cid, inputs=[vid])
        brand.change(cbrand, inputs=[brand])
        position.select(cposition, inputs=[position])
        grade.select(cgrade, inputs=[grade])
        req.change(cmand, inputs=[req])
        add.change(cadd, inputs=[add])
        comp.change(ccomp, inputs=[comp])
        cond.change(ccond, inputs=[cond])
        city.change(ccity, inputs=[city])
        edu.change(cedu, inputs=[edu])
        test_task.change(ctask, inputs=[test_task])
        city.change(ccity, inputs=[city])
        exp.change(cexp, inputs=[exp])
        place.change(cplace, inputs=[place])
        empl.change(cempl, inputs=[empl])
        contr.change(ccontr, inputs=[contr])
        ready.change(submit, inputs=[ready], outputs=[process_button])
        process_button.click(predict, inputs=[process_button], outputs=[output_1, output_2, download_button],
                             show_progress='full')
        download_button.click(save, inputs=[download_button], outputs=[download_button])

    demo.launch(share=True, allowed_paths=[GlobalState.result_dir])


if __name__ == "__main__":
    main()
