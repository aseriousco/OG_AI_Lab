'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
import random
import os, re, logging
import sys
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import pdb
import torch

try:
    import gradio.analytics as analytics
    analytics.version_check = lambda:None
except:...


infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
gpt_path = os.environ.get("gpt_path", None)
sovits_path = os.environ.get("sovits_path", None)
cnhubert_base_path = os.environ.get("cnhubert_base_path", None)
bert_path = os.environ.get("bert_path", None)
version=os.environ.get("version","v2")
        
import gradio as gr
from TTS_infer_pack.TTS import TTS, TTS_Config
from TTS_infer_pack.text_segmentation_method import get_method
from tools.i18n.i18n import I18nAuto, scan_language_list

language=os.environ.get("language","Auto")
language=sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)


# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
# elif torch.backends.mps.is_available():
#     device = "mps"
else:
    device = "cpu"
    
dict_language_v1 = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
}
dict_language_v2 = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("粤语"): "all_yue",#全部按中文识别
    i18n("韩文"): "all_ko",#全部按韩文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("粤英混合"): "yue",#按粤英混合识别####不变
    i18n("韩英混合"): "ko",#按韩英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
    i18n("多语种混合(粤语)"): "auto_yue",#多语种启动切分识别语种
}
dict_language = dict_language_v1 if version =='v1' else dict_language_v2

cut_method = {
    i18n("不切"):"cut0",
    i18n("凑四句一切"): "cut1",
    i18n("凑50字一切"): "cut2",
    i18n("按中文句号。切"): "cut3",
    i18n("按英文句号.切"): "cut4",
    i18n("按标点符号切"): "cut5",
}

tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts_config.device = device
tts_config.is_half = is_half
tts_config.version = version
if gpt_path is not None:
    tts_config.t2s_weights_path = gpt_path
if sovits_path is not None:
    tts_config.vits_weights_path = sovits_path
if cnhubert_base_path is not None:
    tts_config.cnhuhbert_base_path = cnhubert_base_path
if bert_path is not None:
    tts_config.bert_base_path = bert_path
    
print(tts_config)
tts_pipeline = TTS(tts_config)
gpt_path = tts_config.t2s_weights_path
sovits_path = tts_config.vits_weights_path
version = tts_config.version

def inference(text, text_lang, 
              ref_audio_path, 
              aux_ref_audio_paths,
              prompt_text, 
              prompt_lang, top_k, 
              top_p, temperature, 
              text_split_method, batch_size, 
              speed_factor, ref_text_free,
              split_bucket,fragment_interval,
              seed, keep_random, parallel_infer,
              repetition_penalty
              ):

    seed = -1 if keep_random else seed
    actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
    inputs={
        "text": text,
        "text_lang": dict_language[text_lang],
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": [item.name for item in aux_ref_audio_paths] if aux_ref_audio_paths is not None else [],
        "prompt_text": prompt_text if not ref_text_free else "",
        "prompt_lang": dict_language[prompt_lang],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method[text_split_method],
        "batch_size":int(batch_size),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "return_fragment":False,
        "fragment_interval":fragment_interval,
        "seed":actual_seed,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty,
    }
    for item in tts_pipeline.run(inputs):
        yield item, actual_seed
        
def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def change_choices():
    SoVITS_names, GPT_names = get_weights_names(GPT_weight_root, SoVITS_weight_root)
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


pretrained_sovits_name=["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth", "GPT_SoVITS/pretrained_models/s2G488k.pth"]
pretrained_gpt_name=["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"]
_ =[[],[]]
for i in range(2):
    if os.path.exists(pretrained_gpt_name[i]):
        _[0].append(pretrained_gpt_name[i])
    if os.path.exists(pretrained_sovits_name[i]):
        _[-1].append(pretrained_sovits_name[i])
pretrained_gpt_name,pretrained_sovits_name = _

SoVITS_weight_root=["SoVITS_weights_v2","SoVITS_weights"]
GPT_weight_root=["GPT_weights_v2","GPT_weights"]
for path in SoVITS_weight_root+GPT_weight_root:
    os.makedirs(path,exist_ok=True)

def get_weights_names(GPT_weight_root, SoVITS_weight_root):
    SoVITS_names = [i for i in pretrained_sovits_name]
    for path in SoVITS_weight_root:
        for name in os.listdir(path):
            if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (path, name))
    GPT_names = [i for i in pretrained_gpt_name]
    for path in GPT_weight_root:
        for name in os.listdir(path):
            if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (path, name))
    return SoVITS_names, GPT_names


SoVITS_names, GPT_names = get_weights_names(GPT_weight_root, SoVITS_weight_root)



def change_sovits_weights(sovits_path,prompt_language=None,text_language=None):
    tts_pipeline.init_vits_weights(sovits_path)
    global version, dict_language
    dict_language = dict_language_v1 if tts_pipeline.configs.version =='v1' else dict_language_v2
    if prompt_language is not None and text_language is not None:
        if prompt_language in list(dict_language.keys()):
            prompt_text_update, prompt_language_update = {'__type__':'update'},  {'__type__':'update', 'value':prompt_language}
        else:
            prompt_text_update = {'__type__':'update', 'value':''}
            prompt_language_update = {'__type__':'update', 'value':i18n("中文")}
        if text_language in list(dict_language.keys()):
            text_update, text_language_update = {'__type__':'update'}, {'__type__':'update', 'value':text_language}
        else:
            text_update = {'__type__':'update', 'value':''}
            text_language_update = {'__type__':'update', 'value':i18n("中文")}
        return  {'__type__':'update', 'choices':list(dict_language.keys())}, {'__type__':'update', 'choices':list(dict_language.keys())}, prompt_text_update, prompt_language_update, text_update, text_language_update

import os
import base64

# Get the absolute path to the logo
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
logo_path = os.path.join(parent_dir, "static", "A_Serious_Logo.png")

# Read and encode the logo file
with open(logo_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

head_html = """
<head>
    <link rel="shortcut icon" type="image/png" href="data:image/png;base64,%s">
    <link rel="icon" type="image/png" href="data:image/png;base64,%s">
</head>
""" % (encoded_string, encoded_string)

with gr.Blocks(
    title="OG AI Lab",
    css="""
        /* Full viewport background fix */
        :root, html, body {
            background: #032320 !important;
            margin: 0 !important;
            padding: 0 !important;
            min-height: 100vh;
            width: 100vw;
            overflow-x: hidden;
        }

        #root {
            background: #032320 !important;
            width: 100%;
        }

        .gradio-container {
            background: #032320 !important;
            color: white;
            min-height: 100vh;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            margin: 0 !important;
            padding: 0 !important;
            max-width: 100% !important;
            width: 100% !important;
        }

        /* Hide fullscreen button for logo */
        .logo-image > div > div > button {
            display: none !important;
        }

        /* Header with larger logo */
        .header-container {
            border: none !important;
            border-radius: 16px;
            margin: 24px;
            width: calc(100% - 48px) !important;
            overflow: hidden;
            display: flex;
            background: none !important;
            padding: 0 !important;
        }

        .logo-container {
            display: flex;
            align-items: center;
            gap: 24px;
            justify-content: space-between;
            width: 100%;
            background: none !important;
            padding: 0 !important;
        }

        /* Left side with logo */
        .logo-container > div:first-child {
            background: #097A6F !important;
            padding: 24px;
            width: 50%;
            border-radius: 16px;
            display: flex;
            align-items: center;
        }

        /* Right side with title */
        .logo-container > div:last-child {
            background: rgba(48, 48, 48, 1) !important;
            padding: 24px;
            width: 50%;
            border-radius: 16px;
            display: flex;
            align-items: center;
        }

        .logo-container img {
            width: 220px !important;
            height: auto !important;
            object-fit: contain !important;
        }

        /* Hide download and fullscreen buttons for logo */
        .logo-container img + div {
            display: none !important;
        }

        /* Main content area fix */
        .contain {
            max-width: 100% !important;
            padding: 0 !important;
        }

        /* Group Container Styling */
        .group {
            background: rgba(48, 48, 48, 1) !important;
            border: none !important;
            border-radius: 16px;
            padding: 24px;
            margin: 24px;
            width: calc(100% - 48px) !important;
        }

        /* Improved Navigation Tabs */
        .tabs > .tab-nav {
            background: rgba(48, 48, 48, 0.9) !important;
            padding: 15px 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            border: none !important;
            min-height: 60px !important;
        }

        .tabs > .tab-nav > button {
            font-size: 16px !important;
            padding: 12px 25px !important;
            margin: 0 10px !important;
            border-radius: 8px !important;
            background: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border: none !important;
            transition: all 0.3s ease !important;
            min-height: 45px !important;
        }

        .tabs > .tab-nav > button:hover {
            background: rgba(48, 48, 48, 0.8) !important;
            transform: translateY(-2px);
        }

        .tabs > .tab-nav > button.selected {
            background: rgba(48, 48, 48, 1) !important;
        }

        /* Input Elements */
        .gradio-textbox input,
        .gradio-dropdown > select,
        .gradio-slider > div > input {
            background: rgba(3, 35, 32, 0.9) !important;
            border: 1px solid rgba(9, 122, 111, 0.2) !important;
            border-radius: 8px !important;
            color: white !important;
            padding: 10px 15px !important;
        }

        .gradio-textbox textarea {
            background: rgba(3, 35, 32, 0.9) !important;
            border: 1px solid rgba(9, 122, 111, 0.2) !important;
            border-radius: 8px !important;
            color: white !important;
            padding: 12px !important;
        }

        /* Buttons */
        button.primary {
            background: #097A6F !important;
            border: none !important;
            color: white !important;
            padding: 12px 25px !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }

        button.primary:hover {
            transform: translateY(-2px);
            background: #0B8C80 !important;
            box-shadow: 0 4px 12px rgba(9, 122, 111, 0.3) !important;
        }

        button.secondary {
            background: rgba(9, 122, 111, 0.1) !important;
            border: 1px solid rgba(9, 122, 111, 0.2) !important;
            color: white !important;
        }

        button.secondary:hover {
            background: rgba(9, 122, 111, 0.2) !important;
            border-color: #097A6F !important;
        }

        /* Labels */
        label {
            color: rgba(255, 255, 255, 0.9) !important;
            font-weight: 500 !important;
            margin-bottom: 6px !important;
        }

        /* Markdown Text */
        .markdown-text {
            color: white !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
        }

        /* Enhanced Footer */
        .bottom-banner {
            background: linear-gradient(to right, rgba(3, 35, 32, 0.95), rgba(9, 122, 111, 0.95)) !important;
            padding: 25px 0;
            margin: 0;
            text-align: center;
            backdrop-filter: blur(10px);
            width: 100% !important;
            position: relative;
        }

        .bottom-banner::before,
        .bottom-banner::after {
            content: '';
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            width: 85%;
            height: 1px;
            background: linear-gradient(
                to right,
                rgba(255, 255, 255, 0),
                rgba(255, 255, 255, 0.3) 20%,
                rgba(255, 255, 255, 0.3) 80%,
                rgba(255, 255, 255, 0)
            );
        }

        .bottom-banner::before {
            top: 0;
        }

        .bottom-banner::after {
            bottom: 0;
        }

        .bottom-banner-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }

        .bottom-banner-title {
            font-size: 20px;
            font-weight: 600;
            background: linear-gradient(135deg, #097A6F, #0B8C80);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .bottom-banner-subtitle {
            color: rgba(255, 255, 255, 0.7);
            font-size: 15px;
            font-weight: 300;
            letter-spacing: 0.5px;
        }

        /* Additional Interactive Elements */
        input[type="range"] {
            accent-color: #097A6F !important;
        }

        input[type="range"]::-webkit-slider-thumb {
            background: #097A6F !important;
        }

        input[type="range"]::-moz-range-thumb {
            background: #097A6F !important;
        }

        /* Checkbox and Radio buttons */
        input[type="checkbox"], input[type="radio"] {
            accent-color: #097A6F !important;
        }

        /* Selected text */
        ::selection {
            background: #097A6F !important;
            color: white !important;
        }

        /* Focus indicators */
        *:focus {
            outline-color: #097A6F !important;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar-thumb {
            background: #097A6F !important;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #0B8C80 !important;
        }

        /* Loading indicators */
        .loading-spinner {
            border-top-color: #097A6F !important;
        }

        /* Gradio specific elements */
        .gr-button-primary {
            background: #097A6F !important;
            border: none !important;
        }

        .gr-button-secondary {
            color: #097A6F !important;
            border-color: #097A6F !important;
        }

        .gr-button-secondary:hover {
            background: rgba(9, 122, 111, 0.1) !important;
        }

        .gr-form-field:focus-within {
            border-color: #097A6F !important;
        }

        footer {visibility: hidden}
    """
) as app:
    gr.HTML(head_html)
    with gr.Column():
        with gr.Group(elem_classes="header-container"):
            with gr.Row(elem_classes="logo-container"):
                with gr.Column():
                    gr.Image(
                        logo_path,
                        label=None,
                        show_label=False,
                        width=220,
                        container=False,
                        show_download_button=False,
                        interactive=False,
                        elem_classes="logo-image"
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        # OG AI Lab
                        ### 高级语音合成与转换平台
                        """
                    )
    
    with gr.Column():
        # with gr.Group():
        gr.Markdown(value=i18n("模型切换"))
        with gr.Row():
            GPT_dropdown = gr.Dropdown(label=i18n("GPT模型列表"), choices=sorted(GPT_names, key=custom_sort_key), value=gpt_path, interactive=True)
            SoVITS_dropdown = gr.Dropdown(label=i18n("SoVITS模型列表"), choices=sorted(SoVITS_names, key=custom_sort_key), value=sovits_path, interactive=True)
            refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
            refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])

    
    with gr.Row():
        with gr.Column():
            gr.Markdown(value=i18n("*请上传并填写参考信息"))
            with gr.Row():
                inp_ref = gr.Audio(label=i18n("主参考音频(请上传3~10秒内参考音频，超过会报错！)"), type="filepath")
                inp_refs = gr.File(label=i18n("辅参考音频(可选多个，或不选)"),file_count="multiple")
            prompt_text = gr.Textbox(label=i18n("主参考音频的文本"), value="", lines=2)
            with gr.Row():
                prompt_language = gr.Dropdown(
                    label=i18n("主参考音频的语种"), choices=list(dict_language.keys()), value=i18n("中文")
                )
                with gr.Column():
                    ref_text_free = gr.Checkbox(label=i18n("开启无参考文本模式。不填参考文本亦相当于开启。"), value=False, interactive=True, show_label=True)
                    gr.Markdown(i18n("使用无参考文本模式时建议使用微调的GPT，听不清参考音频说的啥(不晓得写啥)可以开，开启后无视填写的参考文本。"))
    
        with gr.Column():
            gr.Markdown(value=i18n("*请填写需要合成的目标文本和语种模式"))
            text = gr.Textbox(label=i18n("需要合成的文本"), value="", lines=20, max_lines=20)
            text_language = gr.Dropdown(
                label=i18n("需要合成的文本的语种"), choices=list(dict_language.keys()), value=i18n("中文")
            )

        
    with gr.Group():
        gr.Markdown(value=i18n("推理设置"))
        with gr.Row():

            with gr.Column():
                batch_size = gr.Slider(minimum=1,maximum=200,step=1,label=i18n("batch_size"),value=20,interactive=True)
                fragment_interval = gr.Slider(minimum=0.01,maximum=1,step=0.01,label=i18n("分段间隔(秒)"),value=0.3,interactive=True)
                speed_factor = gr.Slider(minimum=0.6,maximum=1.65,step=0.05,label="speed_factor",value=1.0,interactive=True)
                top_k = gr.Slider(minimum=1,maximum=100,step=1,label=i18n("top_k"),value=5,interactive=True)
                top_p = gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("top_p"),value=1,interactive=True)
                temperature = gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("temperature"),value=1,interactive=True)
                repetition_penalty = gr.Slider(minimum=0,maximum=2,step=0.05,label=i18n("重复惩罚"),value=1.35,interactive=True)
            with gr.Column():
                with gr.Row():
                    how_to_cut = gr.Dropdown(
                            label=i18n("怎么切"),
                            choices=[i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"), i18n("按中文句号。切"), i18n("按英文句号.切"), i18n("按标点符号切"), ],
                            value=i18n("凑四句一切"),
                            interactive=True, scale=1
                        )
                    parallel_infer = gr.Checkbox(label=i18n("并行推理"), value=True, interactive=True, show_label=True)
                    split_bucket = gr.Checkbox(label=i18n("数据分桶(并行推理时会降低一点计算量)"), value=True, interactive=True, show_label=True)
                
                with gr.Row():  
                    seed = gr.Number(label=i18n("随机种子"),value=-1)
                    keep_random = gr.Checkbox(label=i18n("保持随机"), value=True, interactive=True, show_label=True)

                output = gr.Audio(label=i18n("输出的语音"))
                with gr.Row():
                    inference_button = gr.Button(i18n("合成语音"), variant="primary")
                    stop_infer = gr.Button(i18n("终止合成"), variant="primary")
                
        
        inference_button.click(
            inference,
            [
                text,text_language, inp_ref, inp_refs,
                prompt_text, prompt_language, 
                top_k, top_p, temperature, 
                how_to_cut, batch_size, 
                speed_factor, ref_text_free,
                split_bucket,fragment_interval,
                seed, keep_random, parallel_infer,
                repetition_penalty
             ],
            [output, seed],
        )
        stop_infer.click(tts_pipeline.stop, [], [])
        SoVITS_dropdown.change(change_sovits_weights, [SoVITS_dropdown,prompt_language,text_language], [prompt_language,text_language,prompt_text,prompt_language,text,text_language])
        GPT_dropdown.change(tts_pipeline.init_t2s_weights, [GPT_dropdown], [])

    with gr.Group():
        gr.Markdown(value=i18n("文本切分工具。太长的文本合成出来效果不一定好，所以太长建议先切。合成会根据文本的换行分开合成再拼起来。"))
        with gr.Row():
            text_inp = gr.Textbox(label=i18n("需要合成的切分前文本"), value="", lines=4)
            with gr.Column():
                _how_to_cut = gr.Radio(
                            label=i18n("怎么切"),
                            choices=[i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"), i18n("按中文句号。切"), i18n("按英文句号.切"), i18n("按标点符号切"), ],
                            value=i18n("凑四句一切"),
                            interactive=True,
                        )
                cut_text= gr.Button(i18n("切分"), variant="primary")
            
            def to_cut(text_inp, how_to_cut):
                if len(text_inp.strip()) == 0 or text_inp==[]:
                    return ""
                method = get_method(cut_method[how_to_cut])
                return method(text_inp)
        
            text_opt = gr.Textbox(label=i18n("切分后文本"), value="", lines=4)
            cut_text.click(to_cut, [text_inp, _how_to_cut], [text_opt])
        gr.Markdown(value=i18n("后续将支持转音素、手工修改音素、语音合成分步执行。"))

    gr.Markdown(
        """
        <div class="bottom-banner">
            <div class="bottom-banner-content">
                <div class="bottom-banner-title">Powered by A Serious Company</div>
                <div class="bottom-banner-subtitle">Advancing the Future of Voice Technology</div>
            </div>
        </div>
        """,
        elem_classes="bottom-banner"
    )

app.queue().launch(
    server_name="0.0.0.0",
    server_port=infer_ttswebui,
    share=is_share,
    inbrowser=True,
    quiet=True,
    favicon_path=logo_path
)

if __name__ == '__main__':
    app.queue().launch(#concurrency_count=511, max_size=1022
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=infer_ttswebui,
        show_api=False,
        show_error=False,
        quiet=True,
        favicon_path=logo_path  
    )
