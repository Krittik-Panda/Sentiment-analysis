══════════════════════════════════════════════════════════════
                    COMPLETE PROJECT WORKFLOW
══════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────
 FILE STRUCTURE
────────────────────────────────────────────────────────────

src_2/
├── app.py            ← main Flask app, all routes live here
├── config.py         ← database URL config
├── table.py          ← SQLAlchemy model (Prediction table)
├── preprocess.py     ← text cleaning logic
├── model.pkl         ← trained ML model
├── stop-words-list.txt
├── features/
│   └── vectorizer.pkl
└── templates/
    ├── index.html    ← input form
    ├── result.html   ← shows prediction result
    └── history.html  ← shows all past predictions


────────────────────────────────────────────────────────────
 STARTUP FLOW  (when you run python app.py)
────────────────────────────────────────────────────────────

  app.py starts
      │
      ├── loads Config → gets DATABASE_URL from environment
      ├── db.init_app(app) → connects Flask to SQLAlchemy
      ├── loads model.pkl → ML model ready
      ├── loads vectorizer.pkl → text vectorizer ready
      ├── loads stopwords → stopwords list ready
      │
      └── db.create_all() → creates "prediction" table
                            in PostgreSQL if not exists


────────────────────────────────────────────────────────────
 ROUTE MAP
────────────────────────────────────────────────────────────

  GET  /              → welcome()       → index.html
  POST /predict/      → predict()       → redirects to /result/
  GET  /result/<label>→ result()        → result.html
  GET  /history/      → history_page()  → history.html
  GET  /api/history   → api_history_get()    → JSON
  DELETE /api/history → api_history_delete() → JSON


────────────────────────────────────────────────────────────
 PREDICTION FLOW  (user submits a tweet)
────────────────────────────────────────────────────────────

  User types text → clicks Predict
          │
          ▼
  POST /predict/
          │
          ├── 1. get input_text from form
          │
          ├── 2. clean_text(input_text, stopwords)
          │         removes noise, stopwords, special chars
          │
          ├── 3. vectorizer.transform([clean_input_text])
          │         converts text → numerical array (X)
          │
          ├── 4. if X.nnz == 0
          │         text has no known words
          │         labell = "Text contains no known vocabulary."
          │         → skip ML, go straight to result
          │
          ├── 5. model.predict(X)        → positive / negative
          │    model.predict_proba(X)    → [neg_prob, pos_prob]
          │
          ├── 6. decide label based on probabilities
          │
          │     neg_prob or pos_prob between 0.45–0.55
          │         → "The text is neutral"
          │
          │     neg_prob between 0.50–0.60
          │         → "The text is neutral to negative"
          │
          │     pos_prob between 0.50–0.60
          │         → "The text is neutral to positive"
          │
          │     anything else
          │         → "Text is positive" / "Text is negative"
          │
          ├── 7. save to PostgreSQL
          │         Prediction(
          │             text      = input_text,
          │             label     = labell,
          │             pos_proba = float(probability[0][1]),
          │             neg_proba = float(probability[0][0])
          │         )
          │         db.session.add(data)
          │         db.session.commit()
          │
          └── 8. redirect to /result/<labell>
                      → result.html renders the label


────────────────────────────────────────────────────────────
 HISTORY FLOW  (user clicks See Prediction History)
────────────────────────────────────────────────────────────

  User clicks "See Prediction History"
          │
          ▼
  GET /history/
          │
          └── renders history.html (empty table + spinner)
                      │
                      │  page loads in browser
                      │
                      ▼
              JavaScript runs automatically
                      │
                      ▼
              fetch('GET /api/history')
                      │
                      ▼
              Flask queries PostgreSQL
              Prediction.query
                .order_by(created_at.desc())
                .all()
                      │
                      ▼
              returns JSON
              {
                "data": [
                  {
                    "id": 1,
                    "input_text": "you are a cow",
                    "result": "The text is neutral to positive",
                    "pos_prob": 0.599,
                    "neg_prob": 0.400,
                    "created_at": "2026-05-23T15:25:44"
                  },
                  ...
                ]
              }
                      │
                      ▼
              JavaScript builds table rows
              renders confidence bars
              enables Clear button


────────────────────────────────────────────────────────────
 CLEAR HISTORY FLOW
────────────────────────────────────────────────────────────

  User clicks "Clear Records"
          │
          ▼
  confirm modal appears
          │
  User clicks "Yes, Clear"
          │
          ▼
  fetch('DELETE /api/history')
          │
          ▼
  Flask runs
  Prediction.query.delete()
  db.session.commit()
          │
          ▼
  returns { "message": "History cleared." }
          │
          ▼
  JavaScript calls loadHistory() again
  table shows "No predictions yet."


────────────────────────────────────────────────────────────
 DATABASE TABLE  (PostgreSQL on Render)
────────────────────────────────────────────────────────────

  table name → prediction

  ┌────────────┬──────────────┬───────────────────────┐
  │ column     │ type         │ note                  │
  ├────────────┼──────────────┼───────────────────────┤
  │ id         │ Integer      │ auto increment        │
  │ text       │ String(1000) │ raw input from user   │
  │ label      │ String(50)   │ prediction result     │
  │ pos_proba  │ Float        │ positive confidence   │
  │ neg_proba  │ Float        │ negative confidence   │
  │ created_at │ DateTime     │ auto filled on insert │
  └────────────┴──────────────┴───────────────────────┘


────────────────────────────────────────────────────────────
 CONFIG FLOW
────────────────────────────────────────────────────────────

  config.py
      reads DATABASE_URL from environment variable
      fixes postgres:// → postgresql://
      passes to Flask via app.config.from_object(Config)
          │
          ▼
      Local machine  → set DATABASE_URL = External URL
      Render deploy  → set DATABASE_URL = Internal URL


────────────────────────────────────────────────────────────
 DEPLOYMENT ON RENDER
────────────────────────────────────────────────────────────

  Render PostgreSQL instance
      └── provides Internal Database URL

  Render Web Service
      ├── Build Command  : pip install -r requirements.txt
      ├── Start Command  : python app.py
      └── Environment
              DATABASE_URL = <Internal Database URL>

══════════════════════════════════════════════════════════════