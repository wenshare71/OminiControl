# 远程执行反馈手册(给 MiniMax)

> 你(MiniMax)在远程机器(8×4090)上负责执行 `repro/stage*.ipynb` 实验。
> 本手册规定:notebook 报错时,你该收集什么、按什么格式反馈。
> **你的唯一职责是如实执行和如实汇报,不要自行修改代码、不要自行降级参数重试。**
> 修复方案由本地端(Claude)根据你的反馈决定。

## 一、总原则

1. **原样保留,禁止转述**:traceback、日志、命令输出一律贴原文,不要总结成
   "好像是显存不够了"这类自然语言。转述会丢掉定位问题的关键字段。
2. **禁止截断关键信息**:traceback 要**从第一行到最后一行完整**贴出。
   Jupyter 输出面板显示 "Output is truncated" 时,先点开完整输出再复制,
   或用 §四的命令从 .ipynb 文件里直接提取。
3. **失败即停**:某个 cell 报错后,不要继续执行后面的 cell(后面的 cell 依赖
   前面的状态,连环报错只会制造噪音)。也不要重启 kernel 重试——重试前的现场
   (显存占用、已生成的文件)本身就是证据。
4. **一次报告一份,信息集中**:按 §三的模板把所有信息放进同一份反馈里,
   不要挤牙膏式分多次补充。

## 二、报错后立即收集(按顺序执行)

在**不重启 kernel、不杀进程**的前提下,另开一个终端执行:

```bash
# 1. GPU 现场(哪张卡满了、哪个进程占的,报错后第一时间抓)
nvidia-smi

# 2. 实验产物进度(哪些 case 已完成、断在哪一组)
ls -la repro/subject_lora_compare_512/ 2>/dev/null   # 目录名按实际 OUT_DIR 调整

# 3. 环境指纹(版本不符是常见根因)
source train/setup_env.sh 2>/dev/null
python -c "import torch, diffusers, transformers, peft; \
print('torch', torch.__version__, '| diffusers', diffusers.__version__, \
'| transformers', transformers.__version__, '| peft', peft.__version__)"

# 4. 当前代码版本(本地可能刚推了修复,必须知道你跑的是哪个 commit)
git log --oneline -3 && git status -sb | head -5 && git diff --stat
```

## 三、反馈模板(逐项填写,原文粘贴)

```markdown
## 实验报错反馈

- **notebook**: repro/stageN_xxx.ipynb
- **报错 cell**: 第 N 个代码 cell(贴 cell 开头几行代码以便对齐)
- **报错前进度**: 已完成 X/24 组;最后一条 [DEBUG] run_one 输出原文:
  <粘贴>
- **commit**: <git log --oneline -1 输出>
- **git status**: <是否有未提交的本地改动;git diff --stat 输出>

### 完整 traceback(从第一行到最后一行,不截断)
<粘贴>

### nvidia-smi(报错后立即抓取)
<粘贴>

### 版本指纹
<粘贴 python -c 那条命令的输出>

### 输出目录现状
<粘贴 ls -la 输出>

### 其它异常现象(可选)
如:某 cell 执行特别慢、日志里有 warning、图片肉眼可见异常等,原文贴 warning。
```

## 四、常用取证命令

**从 .ipynb 文件里提取某个 cell 的完整输出**(绕过 Jupyter 前端截断):

```bash
python3 - <<'EOF'
import json
nb = json.load(open("repro/stage7_subject_lora_compare.ipynb"))
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            print(f"===== cell {i} ERROR =====")
            print("\n".join(o.get("traceback", [])))
        elif "text" in o:
            txt = "".join(o["text"])
            if txt.strip():
                print(f"----- cell {i} stdout(末 2000 字符)-----")
                print(txt[-2000:])
EOF
```

**OOM 类报错额外抓**(定位显存去向):

```bash
python3 -c "
import torch
for d in range(torch.cuda.device_count()):
    f, t = torch.cuda.mem_get_info(d)
    print(f'cuda:{d} free={f/2**30:.2f}G total={t/2**30:.2f}G')"
```

**卡死/无响应类**(cell 长时间不动,不算报错但也要反馈):

```bash
py_pid=$(pgrep -f ipykernel | head -1)
cat /proc/$py_pid/io          # 隔 30 秒抓两次,对比判断是否还在读盘(Ceph 慢盘坑)
py-spy dump --pid $py_pid 2>/dev/null || echo "py-spy 未安装,跳过"
```

## 五、明确禁止的行为

| 禁止 | 原因 |
|---|---|
| 报错后自行改代码/参数重跑 | 会破坏现场,且掩盖真实根因(如把 TARGET_SIZE 改小"绕过"OOM) |
| 只贴 traceback 最后一行 | 最后一行往往只是症状,根因在中段的调用链里 |
| 重启 kernel 后才抓 nvidia-smi | 显存现场已被清空,失去证据 |
| 跳过报错 cell 继续跑后续 cell | 连环报错制造噪音;records/annotations 会写出脏数据 |
| 删除或覆盖已生成的 PNG/records.json | 部分完成的产物用于判断断点位置 |
| 转述("好像是显存不够") | 定位需要原始字段(哪张卡、几 GiB、哪一行) |

## 六、正常完成时的汇报

跑完全部 cell 也要按此格式汇报一次:

```markdown
## 实验完成汇报
- notebook / commit: <同上>
- 24 组全部完成,[DEBUG] run_one 输出全文: <粘贴>
- comparison_grid.png 已生成(<文件大小>)
- 平均 speedup: <可视化脚本打印的"平均 speedup"行原文>
- annotations.json / side_map.json 已生成(未标注,待人工 GSB)
```
