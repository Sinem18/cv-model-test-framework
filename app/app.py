
import streamlit as st
import cv2
import os
import sys


# ============================================================
# PROJECT PATH
# ============================================================

# app/app.py
# Bir üst klasör = project root

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Project root'u Python path'e ekle
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.dataset_manager import DatasetManager
from src.model_runner import ModelRunner
from src.evaluator import Evaluator
from src.visualizer import visualize_predictions


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CV Model Test Framework",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 CV Model Test Framework")

st.markdown(
    """
    ### Computer Vision Model Testing Dashboard

    YOLO tabanlı nesne tespit modelinin performansını
    otomatik olarak test eder ve sonuçları görselleştirir.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Test Yapılandırması")


iou_threshold = st.sidebar.slider(
    "IoU Eşik Değeri",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)


confidence_threshold = st.sidebar.slider(
    "Güven Eşiği",
    min_value=0.1,
    max_value=0.9,
    value=0.25,
    step=0.05
)


st.sidebar.write(
    f"IoU: **{iou_threshold:.2f}**"
)

st.sidebar.write(
    f"Güvenilirlik: **{confidence_threshold:.2f}**"
)


# ============================================================
# MODEL / DATASET PATH
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "yolo26n.pt"
)


DATASET_PATH = r"C:\Users\hp\datasets\coco8"


# ============================================================
# RUN EVALUATION
# ============================================================

if st.button(
    "🚀 Çalıştırma Değerlendirmesi",
    type="primary"
):

    try:

        # ----------------------------------------------------
        # INITIALIZATION
        # ----------------------------------------------------

        with st.spinner(
            "Model evaluation çalışıyor..."
        ):

            dataset = DatasetManager(
                DATASET_PATH
            )

            model = ModelRunner(
                MODEL_PATH
            )

            evaluator = Evaluator(
                iou_threshold=iou_threshold
            )


            # ------------------------------------------------
            # GET VALIDATION IMAGES
            # ------------------------------------------------

            images = dataset.get_images(
                "val"
            )


            # ------------------------------------------------
            # TOTAL METRICS
            # ------------------------------------------------

            total_tp = 0
            total_fp = 0
            total_fn = 0


            # Her görüntünün sonucu
            image_results = []


            # ------------------------------------------------
            # EVALUATE EACH IMAGE
            # ------------------------------------------------

            for image in images:

                image_path = str(image)


                # --------------------------------------------
                # READ IMAGE
                # --------------------------------------------

                image_data = cv2.imread(
                    image_path
                )


                if image_data is None:

                    st.warning(
                        f"Görüntü okunamadı: {image_path}"
                    )

                    continue


                height, width = image_data.shape[:2]


                # --------------------------------------------
                # MODEL PREDICTION
                # --------------------------------------------

                predictions = model.predict(
                    image_path
                )


                # --------------------------------------------
                # CONFIDENCE FILTER
                # --------------------------------------------

                predictions = [
                    prediction
                    for prediction in predictions
                    if prediction["confidence"]
                    >= confidence_threshold
                ]


                # --------------------------------------------
                # GET GROUND TRUTH LABELS
                # --------------------------------------------

                labels = dataset.get_labels(
                    image
                )


                # --------------------------------------------
                # EVALUATE IMAGE
                # --------------------------------------------

                result = evaluator.evaluate_image(
                    labels,
                    predictions,
                    width,
                    height
                )


                # --------------------------------------------
                # UPDATE TOTAL METRICS
                # --------------------------------------------

                total_tp += result["tp"]
                total_fp += result["fp"]
                total_fn += result["fn"]


                # --------------------------------------------
                # VISUALIZATION
                # --------------------------------------------

                try:

                    visualized_image = visualize_predictions(
                    image_data,
                    labels,
                    predictions
                   )

                except Exception as visualization_error:

                    visualized_image = image_data

                    st.warning(
                        f"Görselleştirme hatası: "
                        f"{visualization_error}"
                    )


                # --------------------------------------------
                # SAVE IMAGE RESULT
                # --------------------------------------------

                image_results.append({

                    "image": image_path,

                    "tp": result["tp"],

                    "fp": result["fp"],

                    "fn": result["fn"],

                    "visualized": visualized_image

                })


            # ------------------------------------------------
            # FINAL METRICS
            # ------------------------------------------------

            precision = evaluator.calculate_precision(
                total_tp,
                total_fp
            )


            recall = evaluator.calculate_recall(
                total_tp,
                total_fn
            )


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        st.success(
            "Değerlendirme tamamlandı!"
        )


        st.divider()


        # ====================================================
        # EVALUATION RESULTS
        # ====================================================

        st.subheader(
            "📊 Değerlendirme Sonuçları"
        )


        col1, col2, col3, col4, col5 = st.columns(5)


        with col1:

            st.metric(
                "Kesinlik",
                f"{precision:.1%}"
            )


        with col2:

            st.metric(
                "Hatırlamak",
                f"{recall:.1%}"
            )


        with col3:

            st.metric(
                "Gerçek Pozitif",
                total_tp
            )


        with col4:

            st.metric(
                "Yanlış Pozitif",
                total_fp
            )


        with col5:

            st.metric(
                "Yanlış Negatif",
                total_fn
            )


        st.divider()


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.subheader(
            "📋 Model Bilgileri"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.write(
                "**Model:** YOLO26n"
            )

            st.write(
                "**Veri kümesi:** COCO8"
            )

            st.write(
                f"**IoU Eşik Değeri:** "
                f"{iou_threshold:.2f}"
            )


        with col2:

            st.write(
                f"**Doğrulama Görselleri:** "
                f"{len(image_results)}"
            )

            st.write(
                f"**Güven Aralığı:** "
                f"{confidence_threshold:.2f}"
            )

            st.write(
                "**Durum:** Değerlendirme tamamlandı"
            )


        st.divider()


        # ====================================================
        # IMAGE RESULTS
        # ====================================================

        st.subheader(
            "🔎 Görüntü Sonuçları"
        )


        # ----------------------------------------------------
        # LOOP THROUGH IMAGES
        # ----------------------------------------------------

        for result in image_results:


            # -----------------------------------------------
            # IMAGE NAME
            # -----------------------------------------------

            image_name = os.path.basename(
                result["image"]
            )


            st.markdown(
                f"## 🔍 {image_name}"
            )


            # -----------------------------------------------
            # IMAGE DISPLAY
            # -----------------------------------------------

            col1, col2 = st.columns(2)


            with col1:

                st.markdown(
                    "**Orijinal Görüntü**"
                )


                original_image = cv2.imread(
                    result["image"]
                )


                if original_image is not None:

                    original_rgb = cv2.cvtColor(
                        original_image,
                        cv2.COLOR_BGR2RGB
                    )


                    st.image(
                        original_rgb,
                        use_container_width=True
                    )


            with col2:

                st.markdown(
                    "**Model Tahminleri**"
                )


                visualized = result["visualized"]


                if visualized is not None:

                    visualized_rgb = cv2.cvtColor(
                        visualized,
                        cv2.COLOR_BGR2RGB
                    )


                    st.image(
                        visualized_rgb,
                        use_container_width=True
                    )


            # -----------------------------------------------
            # IMAGE METRICS
            # -----------------------------------------------

            metric1, metric2, metric3 = st.columns(3)


            with metric1:

                st.metric(
                    "TP",
                    result["tp"]
                )


            with metric2:

                st.metric(
                    "FP",
                    result["fp"]
                )


            with metric3:

                st.metric(
                    "FN",
                    result["fn"]
                )


            st.divider()


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        st.error(
            "Evaluation sırasında bir hata oluştu."
        )

        st.exception(error)