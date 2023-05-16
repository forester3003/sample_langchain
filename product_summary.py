# 製品名とどの機能について知りたいか、分析対象URLを入力するとまとめてくれる。
# クエリ内容のところを適宜修正して実行する。
# 実行方法：.\product_summary.py "https://bigid.com/" "BigID" "Main" "False" > answer.txt
# 
# 事前準備：
# - .envファイルにOPENAI_API_KEYという名前でOpenAIのAPIキーを記載すること
#
# その他の特徴：
# - 取得したURLのコンテンツをoutput_{product_name}_{point}.txtファイルに出力する。
# - URLにプログラムからアクセスできなかったときは、手動で対象ページのソースをコピーしたmanual_output_{product_name}_{point}.txtファイルを用意するとそれをもとに処理できる。

import os
import sys

from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.vectorstores import Qdrant

def main():
    # 回答保持用変数
    res = ""
    url = ""
    product_name = ""
    point = ""
    verbose = True
    # 引数に何も指定されてなかったら固定値で実行（実行サンプル）
    # 引数の数が3,4だったら実行する。順番通りかとかの入力チェックはしていない。（面倒なので）
    if len(sys.argv) == 1:
        while True:
            print("引数の指定がありません。次の形で実行してください。※ちゃんと\"\"付けてね！\n[<プログラムファイル名> \"URL\" \"プロダクト名\" \"何の機能が知りたいか\" (これは任意だけど詳細出力したいか \"True/False\")]\nTechtouchについて調べるサンプルを実行しますか？（Y/n）")
            answer = input()
            if answer == "Y":
                url = "https://techtouch.jp/"
                product_name = "Techtouch"
                point = "Main"
                break
            elif answer == "n":
                exit()
            else:
                print("Yまたはnを入力してください。")
    elif len(sys.argv) == 4:
        url = sys.argv[1]
        product_name = sys.argv[2]
        point = sys.argv[3]
    elif len(sys.argv) == 5:
        url = sys.argv[1]
        product_name = sys.argv[2]
        point = sys.argv[3]
        verbose = bool(sys.argv[4])
    else:
        print("引数の指定が間違っています。次の形で実行してください。※ちゃんと\"\"付けてね！\n[<プログラムファイル名> \"URL\" \"プロダクト名\" \"何の機能が知りたいか\" (これは任意だけど詳細出力したいか \"True/False\")]")
        exit()

    # .envファイルの確認
    if os.path.exists('.env'):
        load_dotenv('.env')
        OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
        if OPENAI_API_KEY:
            pass
        else:
            print(".envファイルにAPIキーの指定がありません。")
            exit()
    else:
        print(".envファイルがみつかりません。確認してください。")
        exit()

    # 本丸の実行
    res = product_function_summary(url = url, product_name = product_name, point = point, verbose=verbose)
    # 結果の表示
    if(verbose):
        # Trueならすべてprintしているのでスキップ
        pass
    else:
        # Falseなら結果だけ出力
        print(res)

# verboseをtrueにすると標準出力に途中経過も含めて出力する。
def product_function_summary(url: str, product_name: str, point: str, verbose: bool = True) -> str:
    # クエリ内容
    product_name = product_name
    point = point
    url = url
    question = f"の{point}機能は？"
    query = product_name + question

    # APIキー取得
    load_dotenv('.env')
    OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

    # 解析内容保存用ファイル宣言
    output_filename = f'output_{product_name}_{point}.txt'

    # スクレイピングできなかったとき用のファイル読み込み
    html_doc = ""
    if os.path.exists(f'manual_{output_filename}'):
        with open(f'manual_{output_filename}', "r", encoding="utf-8") as f:
            html_doc = f.read()
        
        # 応答をBeautifulSoupオブジェクトに解析する
        soup = BeautifulSoup(html_doc, "html.parser")

        # すべてのテキストを取得する
        text_contents = soup.get_text()

        # テキストをファイルに書き込む
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(text_contents)

    # 何度もスクレイピングしないように結果をファイルに残しておく&前処理の結果ファイルがあったらスキップ
    if os.path.exists(output_filename):
        pass
    else:
        # 指定されたWebサイトにリクエストを送信する
        response = requests.get(url)

        # ステータスコードが200以外なら処理を終了
        if response.status_code != 200:
            print(f'{response.status_code}:リクエストエラーです。スクレイピングできないサイトのようです。\n[manual_{output_filename}]のファイル名でソースをコピーして再実行してください。')
            sys.exit(0)

        # 応答をBeautifulSoupオブジェクトに解析する
        soup = BeautifulSoup(response.text, "html.parser")

        # すべてのテキストを取得する
        text_contents = soup.get_text()

        # テキストをファイルに書き込む
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(text_contents)

    # 検索元にしたいテキストを読み込ませる
    loader = TextLoader(output_filename, 'utf-8')
    documents = loader.load()
    # documents = text_content # 一度ファイルに吐き出さないならこっち
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    qdrant = Qdrant.from_documents(
        docs, embeddings, 
        location=":memory:",  
        collection_name="my_documents",
    )

    # スコア付きの類似文章の検索（0~2くらいまで使おうかな）
    found_docs = qdrant.similarity_search(query)
    '''
    for i in range(3):
        print(f'{i}番目のコンテンツ：')
        print(found_docs[i].page_content)
    '''
    
    # LLM ラッパーを初期化
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

    # 要約用プロンプトテンプレートの作成
    template_summary = """
    Please summarize the following sentence in approximately 500 words.
    Sentences: {text}
    """

    prompt = PromptTemplate(
        input_variables=["text"],
        template = template_summary,
    )

    # LLM チェーンを作成（LLM ラッパーとプロンプトテンプレートから構成する）
    chain = LLMChain(llm=llm, prompt=prompt)

    # LLMチェーンを実行して上位3件の結果をリストへ追加
    summarize_docs = []
    for i in range(min(len(found_docs), 3)):
        prediction = chain.run(text=found_docs[i].page_content)
        # verboseがTrueなら途中経過を出力
        if verbose:
            print("ChatGPTのサマライズ結果：")
            print(f'{prediction.strip()}\n')
        summarize_docs.append(prediction.strip())

    # 3つのサマリ文章をそのまま連結して一つの文章を作成
    input_text = ""
    for summarize_doc in summarize_docs:
        input_text = input_text + summarize_doc

    # 最終回答プロンプトテンプレートの作成
    template_answer = """
    次の文章をもとに日本語で500文字程度で「{query}」に回答してください。
    なお、出力は例のように箇条書きなどMarkdownを用いてわかりやすく簡潔に記載してください。

    例：
    Techtouchはコーディング不要でシステムの改善が可能なプラットフォームです。TechTouchの主要な機能は以下のとおりです。

    - ユーザーのエクスペリエンスを改善するためのデジタルガイド
    - 繰り返しのタスクを自動化するオートフロー機能
    - 使用状況の分析による非効率を特定する機能

    また、以下のような特徴があります。
    - SFA、経費精算、ERPなど企業内で使用されるシステムの改善に役立てることができる
    - 利用の導入時にナビゲーションを設定することができる新しい形式のシステム教育を提供
    - ユーザーフィードバックの即時反映によるサポートコスト削減やセルフオンボーディングの促進など、ユーザーエクスペリエンスの向上に貢献
    - 内部システムだけでなく、自社開発製品にも実装可能

    これらの機能により、Techtouchは全てのユーザーがどんなシステムでも完全に活用できる世界を目指しています。

    文章：{text}
    """

    prompt = PromptTemplate(
        input_variables=["text", "query"],
        template = template_answer,
    )

    # LLM チェーンを作成（LLM ラッパーとプロンプトテンプレートから構成する）
    chain = LLMChain(llm=llm, prompt=prompt)

    prediction = chain.run(text=input_text, query=query)
    # verboseがTrueなら標準出力へ
    if verbose:
        print(f'「{query}」に対するChatGPTの回答：')
        print(f'{prediction.strip()}\n')

    return prediction.strip()

if __name__ == "__main__":
    main()