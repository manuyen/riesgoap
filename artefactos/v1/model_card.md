# Model Card — CAT
**Versión:** v1  
**Entorno:** Python 3.12.7 | scikit-learn 1.5.1

## Datos
Archivo: `cep_aportantes_dataset_para_modelo.csv`  
Shape: (28802, 11)  
Objetivo: `nivel_riesgo` (BAJO=0, ALTO=1)  
Prevalencia (ALTO=1) — TRAIN: 0.885 | TEST: 0.885

## Entrenamiento
Split 80/20 estratificado (random_state=42).  
Preprocesamiento: StandardScaler (num) + OneHotEncoder(ignore) (cat) + SMOTE(k=3).

## Modelo seleccionado
**CAT**  
Umbral de decisión (Paso 8, CV-TRAIN): **0.46**.

## Métricas en TEST (umbral aplicado)
ACC=0.804 | BALACC=0.614 | PREC=0.913 | REC=0.860 | F1=0.886 | MCC=0.196  
ROC-AUC=0.720 | PR-AUC=0.948

## Artefactos
- `pipeline_CAT.joblib`
- `input_schema.json`
- `label_map.json`
- `decision_policy.json`
