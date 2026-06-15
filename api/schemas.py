from pydantic import BaseModel, Field


class CreditApplicationInput(BaseModel):
    revolving_utilization: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio de utilización de crédito rotativo (0–1)",
        examples=[0.45],
    )
    age: int = Field(..., ge=18, le=100, description="Edad del solicitante", examples=[35])
    times_30_59_days_late: int = Field(
        ..., ge=0, description="Veces con mora de 30–59 días", examples=[0]
    )
    debt_ratio: float = Field(
        ..., ge=0.0, description="Ratio de deuda sobre ingresos", examples=[0.25]
    )
    monthly_income: float = Field(
        ..., gt=0, description="Ingreso mensual en USD", examples=[5000.0]
    )
    open_credit_lines: int = Field(
        ..., ge=0, description="Líneas de crédito abiertas", examples=[4]
    )
    times_90_days_late: int = Field(
        ..., ge=0, description="Veces con mora de 90+ días", examples=[0]
    )
    real_estate_loans: int = Field(
        ..., ge=0, description="Préstamos hipotecarios activos", examples=[1]
    )
    times_60_89_days_late: int = Field(
        ..., ge=0, description="Veces con mora de 60–89 días", examples=[0]
    )
    dependents: int = Field(..., ge=0, description="Número de dependientes", examples=[2])


class ShapFeature(BaseModel):
    feature: str
    impact: float


class ScoreResponse(BaseModel):
    probability_of_default: float = Field(
        ..., description="Probabilidad de incumplimiento (0–1)", examples=[0.23]
    )
    risk_level: str = Field(
        ..., description="Nivel de riesgo: Bajo | Medio | Alto", examples=["Bajo"]
    )
    shap_explanation: list[ShapFeature] = Field(
        ..., description="Top 3 variables que influyeron en la decisión"
    )


class ModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    version: str
    stage: str
    algorithm: str | None = Field(
        None,
        description="Algoritmo del modelo: LogisticRegression, XGBClassifier, LGBMClassifier, etc.",
    )
    auc_roc: float | None = Field(
        None, description="AUC-ROC en test set (0.5 = aleatorio, 1.0 = perfecto)"
    )
    ks_statistic: float | None = Field(
        None, description="Kolmogorov-Smirnov: separación entre buenos y malos pagadores"
    )
    gini: float | None = Field(None, description="Coeficiente de Gini: 2 * AUC - 1")
    registered_at: str
