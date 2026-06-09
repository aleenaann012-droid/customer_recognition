from insightface.app import FaceAnalysis

def load_face_model():

    app = FaceAnalysis(name="buffalo_s")

    app.prepare(
        ctx_id=0,
        det_thresh=0.4
    )

    return app
app = load_face_model()
