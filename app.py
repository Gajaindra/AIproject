import os
from flask import Flask, render_template, request
from qa_chain import extract_text_from_pdf, create_qa_chain_from_text

app = Flask(__name__)
qa_chain = None

@app.route("/", methods=["GET", "POST"])
def index():
    global qa_chain
    answer = None

    if request.method == "POST":
        if 'pdf' in request.files:
            file = request.files['pdf']
            text = extract_text_from_pdf(file)
            qa_chain = create_qa_chain_from_text(text)
        elif 'question' in request.form and qa_chain:
            question = request.form['question']
            answer = qa_chain.run(question)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT provided by Render
    app.run(host="0.0.0.0", port=port, debug=True)
