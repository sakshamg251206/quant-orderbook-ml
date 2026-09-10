import sys
from pathlib import Path

import config
from src.logger import logger
from src.explainability import SHAPExplainer


def main():
    logger.info("Initializing SHAP Model Explainability Pipeline...")

    explainer = SHAPExplainer(
        model_path=config.MODELS_DIR / "catboost_1s.pkl",
        processed_dir=config.DATA_DIR / "processed",
        plots_dir=config.DATA_DIR.parent / "plots",
    )

    try:
        # 1. Load model and test dataset
        explainer.load_model_and_data()

        # 2. Compute SHAP values & save shap_values.npy
        explainer.compute_shap_values()

        # 3. Analyze feature importance & save feature_importance.json
        sorted_importance = explainer.analyze_feature_importance()

        # 4. Generate SHAP visualizations
        p_summary = explainer.plot_summary()
        p_importance = explainer.plot_importance()

        p_dep_obi = explainer.plot_dependence("obi_level_5", "shap_dependence_obi.png")
        p_dep_spread = explainer.plot_dependence("spread_relative", "shap_dependence_spread.png")
        p_dep_depth = explainer.plot_dependence("bid_ask_volume_ratio", "shap_dependence_depth.png")

        logger.info("=" * 65)
        logger.info("         SHAP EXPLAINABILITY EXPORT SUMMARY                   ")
        logger.info("=" * 65)
        logger.info(f"SHAP Values Matrix File : {explainer.processed_dir / 'shap_values.npy'}")
        logger.info(f"Feature Importance JSON  : {explainer.processed_dir / 'feature_importance.json'}")
        logger.info("Exported Visualization Plots:")
        logger.info(f"  • Summary Plot        : {p_summary}")
        logger.info(f"  • Bar Importance Plot : {p_importance}")
        logger.info(f"  • Dependence (OBI)    : {p_dep_obi}")
        logger.info(f"  • Dependence (Spread) : {p_dep_spread}")
        logger.info(f"  • Dependence (Depth)  : {p_dep_depth}")
        logger.info("=" * 65)

    except Exception as e:
        logger.error(f"SHAP explainability pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
