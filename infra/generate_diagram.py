"""
Genera el diagrama de arquitectura AWS de CostForecast AI como SVG.

Requisitos previos:
    pip install diagrams
    # Windows: instala Graphviz desde https://graphviz.org/download/ y agrega al PATH
    # macOS:   brew install graphviz
    # Ubuntu:  sudo apt install graphviz

Uso:
    python infra/generate_diagram.py
    # → genera infra/costforecast_aws.svg
"""

from __future__ import annotations

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.compute import ECS, Fargate, Lambda
from diagrams.aws.management import Cloudwatch
from diagrams.aws.ml import Sagemaker, SagemakerModel, SagemakerTrainingJob
from diagrams.aws.network import APIGateway, CloudFront, ALB
from diagrams.aws.security import IAM, SecretsManager
from diagrams.aws.storage import S3
from diagrams.aws.compute import ECR
from diagrams.onprem.client import Users

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

OUT_DIR = Path(__file__).parent
OUT_FILE = str(OUT_DIR / "costforecast_aws")  # diagrams appends the extension

# ---------------------------------------------------------------------------
# Graph attributes
# ---------------------------------------------------------------------------

GRAPH_ATTR = {
    "fontsize": "13",
    "bgcolor": "white",
    "pad": "0.5",
    "splines": "ortho",
    "nodesep": "0.6",
    "ranksep": "0.8",
}

NODE_ATTR = {
    "fontsize": "11",
    "fontname": "Helvetica",
}

EDGE_ATTR = {
    "fontsize": "9",
    "fontname": "Helvetica",
    "color": "#6B7280",
}

# ---------------------------------------------------------------------------
# Diagram
# ---------------------------------------------------------------------------

with Diagram(
    "CostForecast AI — Arquitectura AWS",
    filename=OUT_FILE,
    outformat="svg",
    show=False,
    direction="TB",
    graph_attr=GRAPH_ATTR,
    node_attr=NODE_ATTR,
    edge_attr=EDGE_ATTR,
):
    # ── Actores externos ────────────────────────────────────────────────────
    user = Users("Usuarios")

    # ── Edge ────────────────────────────────────────────────────────────────
    with Cluster("Edge / Acceso público"):
        cdn = CloudFront("CloudFront\n+ WAF")
        apigw = APIGateway("API Gateway\n(REST)")
        alb = ALB("Load Balancer")

    # ── VPC: Aplicación ─────────────────────────────────────────────────────
    with Cluster("VPC  us-east-1"):
        with Cluster("Capa de Aplicación  (ECS Fargate)"):
            streamlit = ECS("Streamlit\nPort 8501")
            fastapi = ECS("FastAPI\nPort 8000")

        # ── Agente ──────────────────────────────────────────────────────────
        with Cluster("Agente IA  (Lambda)"):
            agent = Lambda("LangGraph\nReAct Agent")

        # ── ML Inference ─────────────────────────────────────────────────────
        with Cluster("Inferencia ML  (SageMaker Endpoints)"):
            sm_sar1 = SagemakerModel("SARIMAX\nEquipo 1")
            sm_sar2 = SagemakerModel("SARIMAX\nEquipo 2")
            sm_xgb1 = SagemakerModel("XGBoost\nEquipo 1")
            sm_xgb2 = SagemakerModel("XGBoost\nEquipo 2")

    # ── Datos ────────────────────────────────────────────────────────────────
    with Cluster("Capa de Datos"):
        s3_raw = S3("S3\n/raw  CSVs")
        s3_proc = S3("S3\n/processed\nParquet")
        s3_models = S3("S3\n/models\nArtifacts")
        s3_fc = S3("S3\n/forecasts\nJSON")
        sm_train = SagemakerTrainingJob("SageMaker\nTraining Jobs")

    # ── Infraestructura transversal ──────────────────────────────────────────
    with Cluster("Infraestructura transversal"):
        ecr = ECR("ECR\nImages")
        secrets = SecretsManager("Secrets Manager\nAPI Keys")
        cw = Cloudwatch("CloudWatch\nLogs · Metrics")
        iam = IAM("IAM\nRoles")

    # ── APIs externas ────────────────────────────────────────────────────────
    # Represented as generic nodes (no native Anthropic/Tavily icon)
    with Cluster("APIs Externas"):
        anthropic = Lambda("Anthropic\nClaude API")
        tavily = Lambda("Tavily\nSearch API")

    # ── Conexiones: flujo de usuario ────────────────────────────────────────
    user >> Edge(label="HTTPS /app") >> cdn
    user >> Edge(label="HTTPS /api") >> apigw
    cdn >> alb >> streamlit
    apigw >> fastapi
    streamlit >> Edge(label="REST") >> fastapi

    # ── Conexiones: inferencia ───────────────────────────────────────────────
    fastapi >> Edge(label="forecast") >> agent
    fastapi >> Edge(label="predict") >> [sm_sar1, sm_sar2, sm_xgb1, sm_xgb2]
    agent >> Edge(label="predict") >> [sm_sar1, sm_sar2]
    agent >> Edge(label="shap") >> [sm_xgb1, sm_xgb2]
    fastapi >> Edge(label="write") >> s3_fc

    # ── Conexiones: agente → APIs externas ───────────────────────────────────
    agent >> Edge(label="chat") >> anthropic
    agent >> Edge(label="search") >> tavily

    # ── Conexiones: datos / entrenamiento ────────────────────────────────────
    s3_raw >> Edge(label="ETL") >> s3_proc
    s3_proc >> Edge(label="train_data") >> sm_train
    sm_train >> Edge(label="artifacts") >> s3_models
    s3_models >> Edge(label="load model", style="dashed") >> [sm_sar1, sm_sar2, sm_xgb1, sm_xgb2]

    # ── Conexiones: infraestructura transversal (líneas punteadas) ────────────
    ecr >> Edge(label="pull image", style="dashed", color="#10B981") >> streamlit
    ecr >> Edge(label="pull image", style="dashed", color="#10B981") >> fastapi
    secrets >> Edge(label="read secrets", style="dashed", color="#F59E0B") >> agent
    cw >> Edge(label="monitor", style="dashed", color="#8B5CF6") >> [fastapi, agent]
    iam >> Edge(label="assume role", style="dashed", color="#EF4444") >> [agent, sm_train]


print(f"Diagrama generado: {OUT_FILE}.svg")
