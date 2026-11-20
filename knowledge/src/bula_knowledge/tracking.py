"""MLflow: tracing do workflow e destino das avaliações.

Local por padrão (SQLite em DATA_DIR). Para um workspace do Databricks bastaria
MLFLOW_TRACKING_URI=databricks com as credenciais dele no ambiente; esse caminho
não foi verificado neste projeto.
"""

from __future__ import annotations

import logging

from bula_knowledge.config import Settings

log = logging.getLogger("bula.knowledge.tracking")


def configure_mlflow(settings: Settings) -> None:
    import mlflow

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.effective_mlflow_uri)
    local = settings.mlflow_tracking_uri is None
    if local and mlflow.get_experiment_by_name(settings.mlflow_experiment) is None:
        # No modo local os artefatos ficam junto do banco, em DATA_DIR, e não numa pasta
        # `mlruns` criada onde o comando foi executado.
        artifacts = (settings.data_dir / "mlartifacts").resolve().as_uri()
        mlflow.create_experiment(settings.mlflow_experiment, artifact_location=artifacts)
    mlflow.set_experiment(settings.mlflow_experiment)


def setup_tracing(settings: Settings) -> bool:
    """Liga o tracing automático de LangChain e LangGraph. Devolve se ficou ligado."""
    if not settings.mlflow_tracing:
        return False
    try:
        import mlflow

        configure_mlflow(settings)
        mlflow.langchain.autolog()
    except Exception:
        # Observabilidade não pode impedir o serviço de subir.
        log.warning("tracing do MLflow indisponível; seguindo sem ele", exc_info=True)
        return False
    log.info("tracing do MLflow ligado em %s", settings.effective_mlflow_uri)
    return True
