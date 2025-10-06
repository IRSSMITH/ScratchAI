# core_bot.py (sketch)
import requests, time, sqlite3
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

SCRATCH_USER = "sortwellike"
PROJECT_ID = 1223328801
SESSION = os.environ.get("SCRATCH_SESS")  # set this on host
HEADERS = {"Cookie": f"scratchsessionsid={SESSION};", "X-Requested-With":"XMLHttpRequest","Referer":"https://scratch.mit.edu"}

DB = Path("bot.db")

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS comments (cid INTEGER PRIMARY KEY, author TEXT, content TEXT, replied INTEGER DEFAULT 0)""")
    c.execute("""CREATE TABLE IF NOT EXISTS examples (content TEXT, intent TEXT, reply TEXT)""")
    conn.commit()
    conn.close()

def fetch_comments():
    url = f"https://api.scratch.mit.edu/projects/{PROJECT_ID}/comments/"
    r = requests.get(url)
    return r.json() if r.status_code==200 else []

def post_comment(content, parent_id=None):
    url = f"https://scratch.mit.edu/site-api/comments/project/{PROJECT_ID}/add/"
    data = {"content": content}
    if parent_id:
        data["parent_id"] = parent_id
    r = requests.post(url, headers=HEADERS, data=data)
    return r.status_code==200

# Tiny classifier wrapper (train periodically)
def train_model():
    conn = sqlite3.connect(DB); c=conn.cursor()
    rows = c.execute("SELECT content, intent FROM examples").fetchall()
    conn.close()
    if not rows: return None
    texts, intents = zip(*rows)
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, intents)
    with open("model.pkl","wb") as f:
        pickle.dump((vec, clf), f)

def predict_intent(text):
    import pickle
    try:
        with open("model.pkl","rb") as f:
            vec, clf = pickle.load(f)
    except:
        return "generic"
    X = vec.transform([text])
    return clf.predict(X)[0]

# main loop
def main_loop():
    init_db()
    while True:
        comments = fetch_comments()
        conn = sqlite3.connect(DB); c=conn.cursor()
        for cm in comments:
            cid = cm['id']
            ctext = cm['content']
            c.execute("SELECT 1 FROM comments WHERE cid=?", (cid,))
            if c.fetchone(): continue
            c.execute("INSERT INTO comments (cid, author, content) VALUES (?,?,?)",(cid,cm['author']['username'],ctext))
            intent = predict_intent(ctext.lower())
            # choose reply template
            if intent=="greeting": reply = "Hey! Thanks for stopping by 👋"
            elif intent=="praise": reply = "Thanks so much! That means a lot 😊"
            elif intent=="question": reply = "Nice question — I'll reply properly soon!"
            else: reply = "Thanks for the comment!"
            success = post_comment(reply, parent_id=cid)
            if success:
                c.execute("UPDATE comments SET replied=1 WHERE cid=?", (cid,))
                c.execute("INSERT INTO examples (content, intent, reply) VALUES (?,?,?)",(ctext,intent,reply))
        conn.commit(); conn.close()
        time.sleep(60)

if __name__ == "__main__":
    main_loop()
