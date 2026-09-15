import numpy as np

from app.embeddings import TfidfEmbedder, get_embedder


def test_fit_returns_matrix_with_expected_shape():
    embedder = TfidfEmbedder()
    texts = ["the cat sat on the mat", "dogs are great pets", "cats and dogs"]
    matrix = embedder.fit(texts)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] > 0


def test_embed_query_matches_vocabulary_dimensionality():
    embedder = TfidfEmbedder()
    texts = ["vacation policy for employees", "expense reimbursement rules"]
    matrix = embedder.fit(texts)
    query_vec = embedder.embed_query("how much vacation do employees get")
    assert query_vec.shape[0] == matrix.shape[1]


def test_similar_text_scores_higher_than_unrelated_text():
    embedder = TfidfEmbedder()
    texts = [
        "employees accrue fifteen vacation days per year",
        "airfare must be booked in economy class",
    ]
    matrix = embedder.fit(texts)
    query_vec = embedder.embed_query("how many vacation days do employees get")

    def cosine(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-10
        return float(a @ b) / denom

    sim_vacation = cosine(matrix[0], query_vec)
    sim_airfare = cosine(matrix[1], query_vec)
    assert sim_vacation > sim_airfare


def test_serialization_roundtrip_preserves_vocabulary():
    embedder = TfidfEmbedder()
    embedder.fit(["remote work policy", "onboarding process for new hires"])
    blob = embedder.to_bytes()

    restored = TfidfEmbedder.from_bytes(blob)
    v1 = embedder.embed_query("remote work")
    v2 = restored.embed_query("remote work")
    assert np.allclose(v1, v2)


def test_get_embedder_rejects_unknown_backend():
    try:
        get_embedder("not-a-real-backend")
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_fit_returns_matrix_with_expected_shape()
    test_embed_query_matches_vocabulary_dimensionality()
    test_similar_text_scores_higher_than_unrelated_text()
    test_serialization_roundtrip_preserves_vocabulary()
    test_get_embedder_rejects_unknown_backend()
    print("ok")
