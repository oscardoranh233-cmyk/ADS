# Final adsorption model — 125 features

本仓库已更新为 **2026-11-05 最终确认的 125 特征版本**：

```text
prediction = 0.626 × LightGBMSingle10_BAG_L1 + 0.374 × WeightedEnsemble_L2
model_id   = resin125_fixed_blend_0p626_0p374
```

包含独立运行的特征处理、训练、预测、评估、SMILES 描述符生成、完整融合模型 SHAP 解释和测试代码。代码不依赖上级项目的 Python 模块。模型适用范围为研究中的八种已知树脂。

## 最终模型配置

| 项目 | 基线分量 | 多模型集成分量 |
| --- | --- | --- |
| 输入 | 相同的 125 个特征 | 相同的 125 个特征 |
| 训练数据 | Train + Valid，共 616 条 | Train，共 479 条 |
| 算法 | LightGBM | AutoGluon WeightedEnsemble_L2 |
| Bagging | 单次 10 折 | 单次 10 折；RF/XT 使用引擎的原生 bagging 实现 |
| 随机种子 | 0 | 986 |
| 外层权重 | 0.626 | 0.374 |

保留 7 个基本树脂属性和 9 个树脂–分子交互特征。相对旧版 136 特征，移除了 4 个极性 one-hot、6 个形态衍生特征和 `resin_profile_category`；此前排除的 `resin_surface_x_MolLogP` 仍不使用。精确顺序见 [feature_columns_125.json](config/feature_columns_125.json)。输入中的标签、ID 和其他多余列不会传入模型。

第二分量实际选中 LightGBMXT、RandomForest 和 XGBoost，内部权重为 0.36363636、0.40909091、0.22727273。训练候选还包含 ExtraTrees、CatBoost 和 NeuralNetTorch；完整超参数见 [training_config.json](config/training_config.json)。

## 环境

已保存的模型来自 **Windows、Python 3.9.25、AutoGluon 1.4.0**。请使用匹配环境加载这些模型；完整关键版本见 [training_environment.json](config/training_environment.json)。普通 Python 3.13 环境只用于不加载模型的单元测试。

```powershell
conda create -n adsorption125 python=3.9.25
conda activate adsorption125
python -m pip install -r requirements.txt
```

模型文件使用 pickle，加载本项目可信的模型文件。依赖文件锁定关键直接依赖；它不是所有传递依赖的完整环境锁文件。不同平台或重新训练可能出现数值差异，原始模型的 SHA-256 校验清单用于确认最终版本。

## 本地模型与数据

本地 `models/` 已放入两套确认模型，`data/local/` 已放入锁定划分的三个 CSV。**GitHub 上传代码、配置、校验清单和验证报告；模型二进制、原始研究数据及逐行预测不纳入 Git。** 新克隆需要自行取得这两类文件，才能预测或重新训练。

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

可用 `--model-dir` 指向包含两套预测器的新模型目录。

## 评估与复现

```powershell
python scripts/evaluate.py --verify-artifacts --check-reference
```

该命令核对 100 个模型文件、锁定数据及 479/137/72 划分，再输出 `outputs/evaluation/` 下的指标与逐行预测。

| 口径 | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Train，论文使用的拟合预测 | 0.7309 | 1.5344 | 0.9676 |
| Train，历史混合诊断 | 1.5685 | 2.7713 | 0.8944 |
| Valid，历史诊断 | 2.0752 | 2.9863 | 0.8834 |
| Test，化合物不重叠 | 2.0853 | 2.8427 | 0.7719 |

历史 Train 诊断组合“基线 development OOF + 第二分量 Train 拟合预测”；历史 Valid 诊断组合“基线 development OOF + 第二分量 Valid 推理”。直接调用 `predict()` 对两个分量均使用正常推理，因此 Train/Valid 结果与历史诊断不同。论文使用拟合 Train 和历史 Valid/Test；评估文件明确提供 `manuscript_prediction` 列。拟合 Train 指标偏乐观，且固定 Test 在最终启用之前已被查看，因此报告的 Test 是描述性指标，不是无偏模型选择估计。

重新训练时使用新的输出目录，保留原模型：

```powershell
python scripts/train.py --data-dir data/local --model-dir outputs/retrained_models --output-dir outputs/retrained
```

训练脚本保持原始行顺序、种子、超参数、单次 10 折和固定融合权重，不重新划分数据或搜索权重。数据文件必须匹配锁定校验清单。重新训练不会自动替换确认模型。

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

## 完整融合模型 SHAP（可选）

```powershell
python -m pip install -r requirements-explain.txt
python scripts/explain.py
```

展开 WeightedEnsemble 内部权重，对树模型的子模型分别计算 TreeSHAP 后平均，再按有效权重合并。输出 `outputs/shap/importance.csv`、`shap_values.npz` 和 `audit.json`。解释对象为 616 条 development 数据的直接推理，使用完整融合模型，Test 不参与重要性排序。数值加和校验必须通过；不生成论文排版图。

## 验证

只需 NumPy 和 pandas 即可运行不加载模型的测试：

```powershell
python -m unittest discover -s tests -v
```

也可安装 `requirements-dev.txt` 后运行 `python -m pytest -q`。本次导出完成了 9 项单元测试、真实模型加载、完整划分指标复现、命令行预测以及最终模型 SHAP 验证。具体结果见 [reports/verification.json](reports/verification.json)。完整重新训练未重复执行；保留已确认的模型文件，并验证训练入口的划分、参数与旧配方一致。
