import torch
import gradio as gr
from transformers import AutoTokenizer,  T5ForConditionalGeneration
import sys

# --- 1. 设备和模型配置 ---

# 检查 Apple Silicon (M2 Pro) GPU (MPS) 是否可用
if torch.backends.mps.is_available():
    device = torch.device("mps") # usually would use mps, but for diagnostics we set it explicitly to cpu
    print("M2 Pro (MPS) GPU 准备就绪")
else:
    print("未找到 MPS。正在回退到 CPU (速度会慢很多)。")
    device = torch.device("cpu")

# 指定模型。'google/byt5-base' (约 6 亿参数) 是一个很好的起点。
# 你的 32GB 内存可以轻松运行 'google/byt5-large' (12亿) 
# 甚至 'google/byt5-xl' (37亿)
MODEL_NAME = "google/byt5-large"

# --- 2. 加载模型和分词器 (只需执行一次) ---

print(f"Loading: {MODEL_NAME} ...")
#print("这可能需要几分钟，模型文件将被下载到 'model_cache' 文件夹中。")

try:
    # 使用 torch.float16 (半精度) 
    # 这是在 M2 GPU 上获得最佳性能和内存效率的关键
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 # try between 16 and 32
    ).to(device) # <-- 将模型移动到 M2 GPU

    print("Model loaded successfully and moved to GPU!")

except Exception as e:
    print(f"模型加载失败: {e}")
    print("请检查你的网络连接或 HF_HOME 环境变量。")
    sys.exit(1)


# --- 3. 定义推理函数 ---

def run_byt5(input_text):
    """
    Gradio 将调用的主函数
    """
    if not input_text:
        return "请输入一些文本。"

    print(f"正在处理输入: '{input_text[:20]}...'")

    try:
        # 1. 准备输入
        input_ids = tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512 # ByT5 可以处理长序列
        ).input_ids.to(device) # <-- 确保输入数据也在 GPU 上

        # 2. 生成输出
        outputs = model.generate(input_ids = input_ids, 
                                 max_length=512,
                                 num_beams=4,
                                 early_stopping=True,
                                 no_repeat_ngram_size=2,
                                 length_penalty=1.0)

        # 3. 解码
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"生成完毕: '{decoded_output[:100]}...'")
        return decoded_output

    except Exception as e:
        print(f"推理时发生错误: {e}")
        return f"错误: {e}"

# --- 4. 启动 Gradio Web 界面 ---

iface = gr.Interface(
    fn=run_byt5,
    inputs=gr.Textbox(lines=10, label="Input Text"),
    outputs=gr.Textbox(lines=10, label="ByT5 Output"),
    title="Local ByT5 Large (M2 Pro 32GB)",
    description="Run the ByT5 Large model locally on my M2 Pro GPU. ByT5 is a byte-level T5 model, excellent for handling noisy text and multiple languages. ByT5 Large is the uplimit for 32GB M2 Pro.",
)

print("\nGradio Starting the interface...")
print("Please open http://127.0.0.1:7860 in your browser")
iface.launch()