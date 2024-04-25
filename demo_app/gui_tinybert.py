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
                    'emp_brand': '',
                    'mandatory': '',
                    'additional': '',
                    'comp_stages': '',
                    'work_conditions': '',
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
    status, result = analyzer.classify()
    gr.Info(status)
    if result != 'unknown':
        result = GlobalState.conv_classes[int(result[0])]
    out_2 = f'Predicted by vacancy description conversion - {result}'
    GlobalState.data['conversion_class'] = result
    fid = GlobalState.result_file_path
    append_to_json(GlobalState.data, fid)
    GlobalState.data = GlobalState.default_data
    link = GlobalState.result_file_path
    return gr.update(value=out_2), gr.update(link="/file=" + link, visible=True)


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
                    # Input the text description of the position
                    # 👾👾👾 Then press **Run!** 👾👾👾
                    """)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        brand = gr.Textbox(label='Company name', value=None)
                    with gr.Row():
                        vid = gr.Textbox(label='Position id', value=None)
                    with gr.Row():
                        req = gr.Textbox(label='Mandatory')
                with gr.Column():
                    with gr.Row():
                        add = gr.Textbox(label='Additional')
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
                        output_2 = gr.Textbox(label='LLM Result')
                    with gr.Row():
                        download_button = gr.Button("JSON Archive", visible=False)

        brand.change(cbrand, inputs=[brand])
        vid.change(cid, inputs=[vid])
        req.change(cmand, inputs=[req])
        add.change(cadd, inputs=[add])
        comp.change(ccomp, inputs=[comp])
        cond.change(ccond, inputs=[cond])
        ready.change(submit, inputs=[ready], outputs=[process_button])
        process_button.click(predict, inputs=[process_button], outputs=[output_2, download_button],
                             show_progress='full')
        download_button.click(save, inputs=[download_button], outputs=[download_button])

    demo.launch(share=True, allowed_paths=[GlobalState.result_dir])


if __name__ == "__main__":
    main()
