# Final adsorption model — 125 features

当前默认模型为 **单 LightGBM**，在 616 条开发记录上进行一次十折交叉验证训练，预测时对十个折模型取平均。

```text
prediction = LightGBMSingle10_BAG_L1
model_id   = resin125_baseline_lightgbm_dev616_10fold
```

包含独立运行的特征处理、训练、预测、评估、SMILES 描述符生成、SHAP 解释和测试代码。模型适用范围为研究中的八种已知树脂。

## 最终模型配置

| 项目 | LightGBM |
| --- | --- |
| 输入 | 125 个特征 |
| 训练数据 | 616 条开发记录 |
| 算法 | LightGBM，`min_data_in_leaf=6` |
| Bagging | 按记录单次 10 折 |
| 划分种子 | 1 |
| AutoGluon 运行种子 | 0 |

输入包括分子描述符、MACCS 指纹、7 个基本树脂属性和 9 个树脂–分子交互特征，共 125 列。精确顺序见 [feature_columns_125.json](config/feature_columns_125.json)。输入中的标签、ID 和其他多余列不会传入模型。

折归属由锁定行顺序和划分种子确定。完整超参数见 [training_config.json](config/training_config.json)。

## 环境

已保存的模型来自 **Windows、Python 3.9.25、AutoGluon 1.4.0**。请使用匹配环境加载这些模型；原训练环境的关键版本见 [training_environment.json](config/training_environment.json)。`requirements.txt` 仅安装当前 LightGBM 流程需要的直接依赖。

```powershell
conda create -n adsorption125 python=3.9.25
conda activate adsorption125
python -m pip install -r requirements.txt
```

模型文件使用 pickle，加载本项目可信的模型文件。依赖文件锁定关键直接依赖；它不是所有传递依赖的完整环境锁文件。不同平台或重新训练可能出现数值差异，当前重训模型的 SHA-256 校验清单用于确认最终版本。

## 本地模型与数据

本地 `models/single_baseline/` 保存当前预测器，`data/local/` 保存三个锁定 CSV。两个开发数据文件按固定顺序合并为 616 条记录，再执行十折划分；文件名中的 `train`、`valid` 仅表示存储来源，不作为单独的评价集合。测试数据为另外 72 条记录。预测需要模型文件，重新训练需要锁定数据。

GitHub 仓库包含代码、配置、文件校验清单和汇总指标。拟合模型、原始数据、逐行预测及本地历史备份保留在本地，不随代码提交；从 GitHub 克隆后需先导入模型和数据。

如果有原始研究项目，运行：

```powershell
python scripts/import_local_assets.py --source-root "原始研究项目目录"
```

导入过程校验源文件和已有目标文件的 SHA-256，不覆盖内容不同的模型。路径对应关系见 [models/README.md](models/README.md)；训练数据说明见 [data/README.md](data/README.md)。

## 预测

输入 CSV 或 XLSX 应包含训练时使用的分子描述符、MACCS 列以及 7 个原始树脂列。程序自动重算 9 个交互特征，并按固定顺序选择 125 列。

```powershell
python scripts/predict.py --input data/local/test_model_rows.csv --output outputs/predictions.csv
```

输出保留输入列，并追加 `predicted_adsorption_capacity_mg_per_g`。API 用法（在仓库目录）：

```python
import sys
sys.path.insert(0, "src")
import pandas as pd
from final_adsorption_model import FinalAdsorptionModel

model = FinalAdsorptionModel()
prediction = model.predict(pd.read_csv("data/local/test_model_rows.csv"))
```

可用 `--model-dir` 指向包含 `baseline_single10fold/` 的模型目录。

## 评估与复现

```powershell
python scripts/evaluate.py --verify-artifacts --check-reference
```

该命令校验锁定数据和模型文件，并在 `outputs/evaluation/` 输出 `metrics.csv`、`fold_metrics.csv`、`predictions.csv` 和 `evaluation.json`。汇总仅包含下列三行，不再按 7:2:1 分组报告。已核验的逐折分数见 [fold_metrics.csv](reports/fold_metrics.csv)。

| 评价范围 | 记录数 | R² | RMSE (mg/g) | MAE (mg/g) |
| --- | ---: | ---: | ---: | ---: |
| 十折训练集均值 | 554–555 | 0.9880 | 0.8245 | 0.4232 |
| 十折验证集均值 | 61–62 | 0.8741 | 2.9475 | 1.9367 |
| 固定测试集 | 72 | 0.7825 | 2.7757 | 1.9845 |

训练和验证记录数按每折计。先在每个折模型各自的训练与留出记录上评分，再对十折 R²、RMSE、MAE 分别取算术平均值。测试集先对十个折模型的逐行预测取平均，再计算指标。程序同时校验折外预测的行顺序以及测试预测与十折预测均值的一致性。

开发集按记录划分，折间可能有相同化合物，留出折用于早停。固定测试集包含开发集未出现的九个化合物，曾用于早期模型比较。`predictions.csv` 中开发记录统一标记为 `development`，折号为 1–10；测试记录标记为 `test`，折号与 OOF 预测为空。开发集直接推理列用于诊断，不能用来代替逐折验证指标。

重新训练时使用新的输出目录，保留原模型：

```powershell
python scripts/train.py --data-dir data/local --model-dir outputs/retrained_models --output-dir outputs/retrained
```

训练脚本按锁定行顺序和种子生成十折划分，并应用配置中的超参数。训练结果写入指定的新目录。数据文件必须匹配锁定校验清单。

## 从 SMILES 构造新输入（可选）

```powershell
python -m pip install -r requirements-smiles.txt
python scripts/prepare_features.py --compounds data/compounds.csv --resins data/resins.csv --output outputs/features.csv
python scripts/predict.py --input outputs/features.csv --output outputs/new_predictions.csv
```

`compounds.csv` 需要唯一的 `compound_id` 和 `smiles`；`resins.csv` 需要唯一的 `resin` 及以下数值列：

- `树脂比表面积m2/g_mean`
- `平均孔径Å_mean`
- `湿真密度g/ml_mean`
- `湿视密度g/ml_mean`
- `堆积密度mg/ml_mean`
- `含水率%_mean`
- `resin_polarity_code`

应使用研究中的相同树脂属性与极性编码。每个化合物与每个树脂生成一行。描述符计算沿用已有新化合物预测实现：RDKit 2022.9.5、ETKDGv3 seed 0、UFF、Gasteiger 电荷及 MACCS。`rdkit3d_embed_ok` 记录三维构象生成是否成功；失败时 SASA 回退到 LabuteASA。此入口用于新化合物，不重算或替换锁定训练 CSV 的历史描述符。

## LightGBM 模型 SHAP（可选）

```powershell
python -m pip install -r requirements-explain.txt
python scripts/explain.py
```

对十个 LightGBM 折模型计算 TreeSHAP 并按折平均。输出 `outputs/shap/importance.csv`、`shap_values.npz` 和 `audit.json`。解释对象为 616 条开发数据的直接推理。数值加和校验通过后写出文件。

## 验证

安装 `requirements-dev.txt`（NumPy、pandas、scikit-learn 和 pytest）后，可运行不加载模型的测试：

```powershell
python -m unittest discover -s tests -v
```

也可运行 `python -m pytest -q`。当前版本完成了 13 项单元测试、20 个模型文件的哈希核对、十折训练与验证指标复算、测试预测均值核对，以及项目默认模型与独立包的预测一致性核对。独立重训复现了全部逐折指标和 688 条记录的预测，最大预测差异为 0；616 条开发记录的 SHAP 加和校验也已通过。具体结果见 [reports/verification.json](reports/verification.json)。
