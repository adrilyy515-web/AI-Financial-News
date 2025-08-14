import os
import datetime as dt
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Optional: muat .env saat dev
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------- Konfigurasi API Keys -----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY belum diset. Tambahkan di environment.")
if not NEWSAPI_KEY:
    raise RuntimeError("NEWSAPI_KEY belum diset. Tambahkan di environment.")

# ----------- Init Klien Gemini -----------
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-1.5-flash"  # cepat & hemat; ganti ke 1.5-pro untuk analisis lebih dalam

model = genai.GenerativeModel(
    model_name=GEMINI_MODEL_NAME,
    system_instruction=(
        """
        Anda adalah asisten analitik keuangan & investasi yang berhati-hati dan edukatif.
        Batasan penting:
        - Ini bukan nasihat keuangan personal. Tekankan edukasi & alternatif skenario.
        - Sertakan penjelasan risiko, asumsi, dan horizon waktu saat memberi rekomendasi umum.
        - Jika pengguna menyebut tujuan, profil risiko, atau batasan, gunakan untuk mengkontekstualkan jawaban.
        - Jika ada berita yang diikutkan, rangkum inti, dampak ke pasar/emetent/sektor, dan sebutkan tanggal sumber.
        - Hindari kepastian berlebihan; gunakan probabilitas kualitatif (mis. rendah/sedang/tinggi) bila relevan.
        - Gunakan bahasa Indonesia yang jelas dan ringkas.
        """
    ),
)

# ----------- Init NewsAPI -----------
from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# ----------- Skema Request/Response -----------
class InvestorProfile(BaseModel):
    risiko: Optional[str] = Field(
        default=None,
        description="Toleransi risiko: rendah/sedang/tinggi"
    )
    horizon_bulan: Optional[int] = Field(
        default=None, ge=1, description="Horizon investasi dalam bulan"
    )
    fokus: Optional[str] = Field(
        default=None, description="Fokus: mis. saham, obligasi, ETF, kripto, properti"
    )

class AnalyzeRequest(BaseModel):
    pertanyaan: str = Field(..., description="Pertanyaan pengguna tentang keuangan/investasi")
    tickers: Optional[List[str]] = Field(default=None, description="Daftar ticker/emitmen terkait (opsional)")
    kata_kunci: Optional[str] = Field(default=None, description="Kata kunci pencarian berita tambahan")
    hari_kebelakang: int = Field(default=7, ge=1, le=30, description="Seberapa jauh ambil berita (hari)")
    bahasa: str = Field(default="id", description="Kode bahasa NewsAPI, mis. id atau en")
    profil: Optional[InvestorProfile] = None

class NewsQuery(BaseModel):
    q: str
    dari: Optional[str] = None  # YYYY-MM-DD
    sampai: Optional[str] = None  # YYYY-MM-DD
    bahasa: str = Field(default="id")
    halaman: int = Field(default=1, ge=1, le=5)

# ----------- Utilitas -----------

def _fmt_date(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")


def fetch_news(q: str, language: str = "id", from_days: int = 7, page_size: int = 10):
    """Ambil berita dari NewsAPI (everything endpoint)."""
    today = dt.date.today()
    frm = today - dt.timedelta(days=from_days)
    res = newsapi.get_everything(
        q=q,
        language=language,
        from_param=_fmt_date(frm),
        to=_fmt_date(today),
        sort_by="publishedAt",
        page=1,
        page_size=page_size,
    )
    if res.get("status") != "ok":
        raise HTTPException(status_code=502, detail=f"Gagal memuat berita: {res}")
    articles = res.get("articles", [])
    return [
        {
            "judul": a.get("title"),
            "sumber": a.get("source", {}).get("name"),
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
            "ringkas": a.get("description"),
        }
        for a in articles
    ]


def build_news_context(articles: List[dict]) -> str:
    if not articles:
        return "Tidak ada artikel relevan yang ditemukan dalam jangka waktu yang ditentukan."
    lines = ["Artikel Berita Terkait (ringkasan singkat):"]
    for i, a in enumerate(articles, 1):
        when = a.get("publishedAt", "")
        lines.append(
            f"{i}. [{a['judul']}]({a['url']}) — {a['sumber']} — {when}\n   Ringkas: {a.get('ringkas') or '-'}"
        )
    return "\n".join(lines)


def build_prompt(user_q: str, profile: Optional[InvestorProfile], news_ctx: str) -> List[dict]:
    # Gunakan format "content parts" untuk memberi konteks terstruktur
    parts = []
    if profile:
        parts.append(
            {
                "text": (
                    "Profil Investor:\n"
                    f"- Risiko: {profile.risiko or '-'}\n"
                    f"- Horizon (bulan): {profile.horizon_bulan or '-'}\n"
                    f"- Fokus: {profile.fokus or '-'}\n"
                )
            }
        )
    parts.append({"text": news_ctx})
    parts.append({
        "text": (
            "Tugas:\n"
            "- Jawab pertanyaan pengguna di bawah ini.\n"
            "- Gunakan konteks berita di atas jika relevan.\n"
            "- Sertakan langkah analisis ringkas, poin risiko, dan opsi alternatif.\n"
            "- Akhiri dengan ringkasan eksekutif (3–5 poin bullet).\n"
            "- Tambahkan penafian singkat bahwa ini bukan nasihat keuangan personal.\n"
        )
    })
    parts.append({"text": f"Pertanyaan Pengguna: {user_q}"})
    return parts


# ----------- FastAPI App -----------
app = FastAPI(title="AI Pakar Keuangan & Investasi (Gemini + NewsAPI)")


@app.get("/")
def root():
    return {
        "name": "AI Pakar Keuangan & Investasi",
        "model": GEMINI_MODEL_NAME,
        "endpoints": ["/analyze", "/news", "/suggest-questions"],
        "disclaimer": "Informasi bersifat edukatif, bukan nasihat keuangan personal.",
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    # 1) Rangkai query berita
    terms = []
    if req.kata_kunci:
        terms.append(req.kata_kunci)
    if req.tickers:
        terms.extend(req.tickers)
    if not terms:
        # fallback: ambil berita umum ekonomi/market
        terms = ["ekonomi OR pasar saham OR IHSG"]

    q = " OR ".join(terms)

    # 2) Ambil berita
    articles = fetch_news(q=q, language=req.bahasa, from_days=req.hari_kebelakang, page_size=8)
    news_ctx = build_news_context(articles)

    # 3) Bangun prompt & panggil Gemini
    parts = build_prompt(req.pertanyaan, req.profil, news_ctx)

    try:
        resp = model.generate_content(parts)
        teks = resp.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memanggil Gemini: {e}")

    return {
        "query": req.pertanyaan,
        "news_query": q,
        "articles_count": len(articles),
        "answer": teks,
        "disclaimer": "Konten untuk tujuan edukasi. Lakukan riset mandiri & konsultasi penasihat berizin.",
    }


@app.post("/news")
def news(query: NewsQuery):
    # Validasi tanggal
    frm = query.dari
    to = query.sampai
    params = {
        "q": query.q,
        "language": query.bahasa,
        "sort_by": "publishedAt",
        "page": query.halaman,
        "page_size": 20,
    }

    if frm:
        params["from_param"] = frm
    if to:
        params["to"] = to

    res = newsapi.get_everything(**params)
    if res.get("status") != "ok":
        raise HTTPException(status_code=502, detail=f"Gagal memuat berita: {res}")

    articles = [
        {
            "judul": a.get("title"),
            "sumber": a.get("source", {}).get("name"),
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
            "ringkas": a.get("description"),
        }
        for a in res.get("articles", [])
    ]
    return {"total": res.get("totalResults", 0), "articles": articles}


@app.get("/suggest-questions")
def suggest_questions(topik: Optional[str] = None):
    prompt = (
        "Buat 8 pertanyaan tajam seputar keuangan/investasi untuk membantu analisis. "
        "Variasikan dari makro, sektor, emiten, manajemen risiko, dan perencanaan keuangan. "
    )
    if topik:
        prompt += f"Fokus utama: {topik}. "

    try:
        resp = model.generate_content(prompt)
        teks = resp.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memanggil Gemini: {e}")

    return {"suggestions": [s.strip("- •\n ") for s in teks.split("\n") if s.strip()]}
