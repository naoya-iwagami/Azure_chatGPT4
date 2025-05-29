import os  
import json  
import base64  
import threading  
import datetime  
import uuid  
import streamlit as st  
from azure.search.documents import SearchClient  
from azure.core.credentials import AzureKeyCredential  
from azure.core.pipeline.transport import RequestsTransport  
from azure.cosmos import CosmosClient  
from openai import AzureOpenAI  
from PIL import Image  
import certifi  
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions  
from urllib.parse import urlparse, unquote, quote  

# Azure OpenAI settings  
client = AzureOpenAI(  
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")  
)  
  
# Azure Cognitive Search settings  
search_service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")  
search_service_key = os.getenv("AZURE_SEARCH_KEY")  
  
transport = RequestsTransport(verify=certifi.where())  
  
# Azure Cosmos DB settings  
cosmos_endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")  
cosmos_key = os.getenv("AZURE_COSMOS_KEY")  
database_name = "chatdb"  
container_name = "chathistory"  
  
cosmos_client = CosmosClient(cosmos_endpoint, credential=cosmos_key)  
database = cosmos_client.get_database_client(database_name)  
container = database.get_container_client(container_name)  
  
# Azure Blob Storage settings  
blob_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")  
blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)  
  
lock = threading.Lock()  
  
def extract_account_key(connection_string):  
    pairs = [s.split("=", 1) for s in connection_string.split(";") if "=" in s]  
    conn_dict = dict(pairs)  
    return conn_dict.get("AccountKey")  
  
def generate_sas_url(blob_url, file_name):  
    parsed_url = urlparse(blob_url)  
    account_name = parsed_url.netloc.split(".")[0]  
    container_name = parsed_url.path.split("/")[1]  
    blob_name = "/".join(parsed_url.path.split("/")[2:])  
    blob_name = unquote(blob_name)  
    account_key = extract_account_key(blob_connection_string)  
    expiry_time = datetime.datetime.utcnow() + datetime.timedelta(hours=1)  
    content_disposition = f"attachment; filename*=UTF-8''{quote(file_name)}"  
    sas_token = generate_blob_sas(  
        account_name=account_name,  
        container_name=container_name,  
        blob_name=blob_name,  
        account_key=account_key,  
        permission=BlobSasPermissions(read=True),  
        expiry=expiry_time,  
        content_disposition=content_disposition,  
        encoding="utf-8"  
    )  
    blob_name_encoded = quote(blob_name, safe="")  
    sas_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name_encoded}?{sas_token}"  
    return sas_url  
  
def rewrite_query(user_input, recent_messages, system_message=None):  
    prompt = (  
        "あなたは社内情報検索のためのアシスタントです。\n"  
        "以下の会話履歴とユーザーの最新質問をもとに、検索エンジンに適した簡潔で明確な日本語クエリを1つだけ生成してください。\n"  
        "不要な会話表現や雑談は除外し、検索意図を正確に反映したキーワードやフレーズを含めてください。\n"  
        "【会話履歴】\n"  
    )  
    for msg in recent_messages:  
        prompt += f"- {msg}\n"  
    prompt += f"【ユーザー質問】\n{user_input}\n"  
    prompt += "【検索用クエリ】"  
  
    messages = [  
        {"role": "system", "content": system_message or "あなたは有能な検索アシスタントです。"},  
        {"role": "user", "content": prompt}  
    ]  
  
    response = client.chat.completions.create(  
        model="gpt-4.1",  # ←gpt-4.1に固定しました  
        messages=messages,  
        max_completion_tokens=256,  
    )  
    rewritten_query = response.choices[0].message.content.strip()  
    return rewritten_query  
  
def main():  
    # ユーザー情報取得  
    if hasattr(st, "experimental_user"):  
        user = st.experimental_user  
        if user and "email" in user:  
            st.session_state["user_id"] = user["email"]  
        else:  
            st.error("ユーザー情報を取得できませんでした。")  
            st.stop()  
    else:  
        st.error("ユーザー情報を取得できませんでした。")  
        st.stop()  
  
    st.title("Azure OpenAI ChatGPT with Image Upload and RAG")  
  
    # セッション初期化  
    if "sidebar_messages" not in st.session_state:  
        st.session_state["sidebar_messages"] = []  
    if "main_chat_messages" not in st.session_state:  
        st.session_state["main_chat_messages"] = []  
    if "images" not in st.session_state:  
        st.session_state["images"] = []  
    if "current_chat_index" not in st.session_state:  
        st.session_state["current_chat_index"] = None  
    if "show_all_history" not in st.session_state:  
        st.session_state["show_all_history"] = False  
    if "default_system_message" not in st.session_state:  
        st.session_state["default_system_message"] = "あなたは親切なAIアシスタントです。ユーザーの質問に簡潔かつ正確に答えてください。"  
    if "system_message" not in st.session_state:  
        st.session_state["system_message"] = st.session_state["default_system_message"]  
    if "last_search_query" not in st.session_state:  
        st.session_state["last_search_query"] = ""  
    if "past_message_count" not in st.session_state:  
        st.session_state["past_message_count"] = 10  
  
    # サイドバー：モデル選択  
    st.sidebar.header("モデル選択")  
    model_options = ["gpt-4o", "o1", "o4-mini", "o3", "gpt-4.1", "gpt-4.5-preview"]  
    st.session_state["model_to_use"] = st.sidebar.selectbox(  
        "使用するモデルを選択してください",  
        options=model_options,  
        index=0  
    )  
    model_to_use = st.session_state["model_to_use"]  
  
    reasoning_effort = None  
    if model_to_use in ["o1", "o4-mini", "o3"]:  
        reasoning_effort = st.sidebar.radio(  
            "推論レベルを選択してください",  
            options=['low', 'medium', 'high'],  
            index=1  
        ).lower()  
        st.session_state["reasoning_effort"] = reasoning_effort  
    else:  
        st.session_state.pop("reasoning_effort", None)  
  
    # サイドバー：インデックス選択  
    st.sidebar.header("インデックス選択")  
    index_options = {  
        "通常データ": "filetest11",  
        "SANUQIメール": "filetest13",  
        "L8データ": "filetest14",  
        "L8＋製膜データ": "filetest15",  
        "予備１": "filetest16",  
        "予備２": "filetest17",  
        "予備３": "filetest18",
        "品質保証": "quality-assurance"  
    }  
    selected_index_label = st.sidebar.selectbox(  
        "使用するデータを選択してください",  
        list(index_options.keys()),  
        index=0  
    )  
    index_name = index_options[selected_index_label]  
  
    global search_client  
    search_client = SearchClient(  
        endpoint=search_service_endpoint,  
        index_name=index_name,  
        credential=AzureKeyCredential(search_service_key),  
        transport=transport,  
        api_version="2024-07-01"  
    )  
  
    # --- クエリ生成方法の選択 ---  
    st.sidebar.header("検索クエリ生成方法")  
    query_generation_method = st.sidebar.radio(  
        "検索クエリの生成方法を選択してください",  
        options=["シンプル結合", "クエリリライト"],  
        index=0,  
        key="query_generation_method"  
    )  
  
    def encode_image(image_file):  
        return base64.b64encode(image_file.getvalue()).decode("utf-8")  
  
    def save_chat_history():  
        with lock:  
            try:  
                user_id = st.session_state.get("user_id")  
                current_index = st.session_state.get("current_chat_index")  
                if current_index is None:  
                    return  
                current_chat = st.session_state.sidebar_messages[current_index]  
                session_id = current_chat.get("session_id")  
                if session_id:  
                    item = {  
                        "id": session_id,  
                        "user_id": user_id,  
                        "session_id": session_id,  
                        "messages": current_chat.get("messages", []),  
                        "system_message": current_chat.get("system_message", st.session_state.get("default_system_message", "")),  
                        "first_assistant_message": current_chat.get("first_assistant_message", ""),  
                        "timestamp": datetime.datetime.utcnow().isoformat()  
                    }  
                    container.upsert_item(item)  
            except Exception as e:  
                st.error(f"チャット履歴の保存中にエラーが発生しました: {e}")  
  
    def load_chat_history():  
        with lock:  
            user_id = st.session_state.get("user_id")  
            sidebar_messages = []  
            try:  
                query = "SELECT * FROM c WHERE c.user_id = @user_id ORDER BY c.timestamp DESC"  
                parameters = [{"name": "@user_id", "value": user_id}]  
                items = container.query_items(  
                    query=query,  
                    parameters=parameters,  
                    enable_cross_partition_query=True  
                )  
                for item in items:  
                    if "session_id" in item:  
                        session_id = item["session_id"]  
                        chat = {  
                            "session_id": session_id,  
                            "messages": item.get("messages", []),  
                            "system_message": item.get("system_message", st.session_state.get("default_system_message", "")),  
                            "first_assistant_message": item.get("first_assistant_message", "")  
                        }  
                        sidebar_messages.append(chat)  
            except Exception as e:  
                st.error(f"チャット履歴の読み込み中にエラーが発生しました: {e}")  
            return sidebar_messages  
  
    if not st.session_state["sidebar_messages"]:  
        st.session_state["sidebar_messages"] = load_chat_history()  
  
    def start_new_chat():  
        new_session_id = str(uuid.uuid4())  
        new_chat = {  
            "session_id": new_session_id,  
            "messages": [],  
            "first_assistant_message": "",  
            "system_message": st.session_state["default_system_message"]  
        }  
        st.session_state.sidebar_messages.insert(0, new_chat)  
        st.session_state["current_chat_index"] = 0  
        st.session_state.system_message = new_chat["system_message"]  
        st.session_state.main_chat_messages = []  
        st.session_state.images = []  
        st.session_state["last_search_query"] = ""  
  
    if st.session_state["current_chat_index"] is None:  
        start_new_chat()  
  
    def summarize_text(text, max_length=10):  
        return text[:max_length] + "..." if len(text) > max_length else text  
  
    with st.sidebar:  
        st.header("システムメッセージ設定")  
        if st.session_state["current_chat_index"] is not None:  
            current_system_message = st.session_state.sidebar_messages[st.session_state["current_chat_index"]].get(  
                "system_message", st.session_state["default_system_message"]  
            )  
        else:  
            current_system_message = st.session_state["default_system_message"]  
  
        system_message_key = f"system_message_{st.session_state['current_chat_index']}"  
        system_message = st.text_area(  
            "システムメッセージを入力してください",  
            value=current_system_message,  
            height=100,  
            key=system_message_key  
        )  
        if st.session_state["current_chat_index"] is not None:  
            st.session_state.sidebar_messages[st.session_state["current_chat_index"]]["system_message"] = system_message  
            st.session_state.system_message = system_message  
  
        st.header("チャット履歴")  
        max_displayed_history = 5  
        max_total_history = 20  
        sidebar_msgs = [  
            (index, chat)  
            for index, chat in enumerate(st.session_state.sidebar_messages)  
            if chat.get("first_assistant_message")  
        ]  
        if not st.session_state["show_all_history"]:  
            sidebar_msgs = sidebar_msgs[:max_displayed_history]  
        else:  
            sidebar_msgs = sidebar_msgs[:max_total_history]  
        for i, (orig_index, chat) in enumerate(sidebar_msgs):  
            if chat and "first_assistant_message" in chat:  
                keyword = summarize_text(chat["first_assistant_message"])  
                if st.button(keyword, key=f"history_{i}"):  
                    st.session_state["current_chat_index"] = orig_index  
                    st.session_state.main_chat_messages = chat.get("messages", []).copy()  
                    st.session_state.system_message = chat.get("system_message", st.session_state["default_system_message"])  
                    system_message_key = f"system_message_{st.session_state['current_chat_index']}"  
                    if system_message_key in st.session_state:  
                        del st.session_state[system_message_key]  
                    st.rerun()  
  
        if len(st.session_state.sidebar_messages) > max_displayed_history:  
            if st.session_state["show_all_history"]:  
                if st.button("少なく表示"):  
                    st.session_state["show_all_history"] = False  
                    st.rerun()  
            else:  
                if st.button("もっと見る"):  
                    st.session_state["show_all_history"] = True  
                    st.rerun()  
  
        if st.button("新しいチャット"):  
            start_new_chat()  
            st.session_state["show_all_history"] = False  
            st.rerun()  
  
        st.header("画像アップロード")  
        uploaded_files = st.file_uploader(  
            "画像を選択してください", type=["jpg", "jpeg", "png"],  
            key="uploader", accept_multiple_files=True  
        )  
        if uploaded_files:  
            for uploaded_file in uploaded_files:  
                image = Image.open(uploaded_file)  
                encoded_image = encode_image(uploaded_file)  
                if not any(img["name"] == uploaded_file.name for img in st.session_state["images"]):  
                    st.session_state["images"].append({  
                        "image": image,  
                        "encoded": encoded_image,  
                        "name": uploaded_file.name  
                    })  
                    st.success(f"画像 '{uploaded_file.name}' がアップロードされました。")  
  
        def display_images():  
            st.subheader("アップロードされた画像")  
            for idx, img_data in enumerate(st.session_state["images"]):  
                st.image(img_data["image"], caption=img_data["name"], use_container_width=True)  
                if st.button(f"削除 {img_data['name']}", key=f"delete_{idx}"):  
                    st.session_state["images"].pop(idx)  
                    st.rerun()  
  
        display_images()  
  
        past_message_count = st.slider(  
            "過去メッセージの数", min_value=1, max_value=50, value=st.session_state["past_message_count"], key="past_message_count"  
        )  
        st.header("検索設定")  
        search_mode = st.radio(  
            "検索モードを選択してください",  
            options=["セマンティック検索", "ベクトル検索", "ハイブリッド検索"],  
            index=0  
        )  
        topNDocuments = st.slider("取得するドキュメント数", min_value=1, max_value=300, value=5)  
        strictness = st.slider("厳密度 (スコアの閾値)", min_value=0.0, max_value=10.0, value=0.1, step=0.1)  
  
    def keyword_semantic_search(query, topNDocuments=5, strictness=0.1):  
        results = search_client.search(  
            search_text=query,  
            search_fields=["title", "content"],  
            select="title, content, filepath, url",  
            query_type="semantic",  
            semantic_configuration_name="default",  
            query_caption="extractive",  
            query_answer="extractive",  
            top=topNDocuments  
        )  
        results_list = [result for result in results if result.get("@search.score", 0) >= strictness]  
        results_list.sort(key=lambda x: x.get("@search.score", 0), reverse=True)  
        return results_list  
  
    def get_query_embedding(query):  
        embedding_response = client.embeddings.create(  
            model="text-embedding-3-small",  
            input=query  
        )  
        return embedding_response.data[0].embedding  
  
    def keyword_vector_search(query, topNDocuments=5):  
        query_embedding = get_query_embedding(query)  
        vector_query = {  
            "kind": "vector",  
            "vector": query_embedding,  
            "exhaustive": True,  
            "fields": "contentVector",  
            "weight": 0.5,  
            "k": topNDocuments  
        }  
        results = search_client.search(  
            search_text="*",  
            vector_queries=[vector_query],  
            select="title, content, filepath, url"  
        )  
        results_list = list(results)  
        if results_list and "@search.score" in results_list[0]:  
            results_list.sort(key=lambda x: x.get("@search.score", 0), reverse=True)  
        return results_list  
  
    def keyword_search(query, topNDocuments=5):  
        results = search_client.search(  
            search_text=query,  
            search_fields=["title", "content"],  
            select="title, content, filepath, url",  
            query_type="simple",  
            top=topNDocuments  
        )  
        return list(results)  
  
    def hybrid_search(query, topNDocuments=5, strictness=0.1):  
        keyword_results = keyword_search(query, topNDocuments=topNDocuments)  
        semantic_results = keyword_semantic_search(query, topNDocuments=topNDocuments, strictness=strictness)  
        vector_results = keyword_vector_search(query, topNDocuments=topNDocuments)  
        rrf_k = 60  
        fusion_scores = {}  
        fusion_docs = {}  
        for result_list in [keyword_results, semantic_results, vector_results]:  
            for idx, result in enumerate(result_list):  
                doc_id = result.get("filepath") or result.get("title")  
                if not doc_id:  
                    continue  
                contribution = 1 / (rrf_k + (idx + 1))  
                fusion_scores[doc_id] = fusion_scores.get(doc_id, 0) + contribution  
                if doc_id not in fusion_docs:  
                    fusion_docs[doc_id] = result  
        sorted_doc_ids = sorted(fusion_scores, key=lambda d: fusion_scores[d], reverse=True)  
        fused_results = []  
        for doc_id in sorted_doc_ids[:topNDocuments]:  
            result = fusion_docs[doc_id]  
            result["fusion_score"] = fusion_scores[doc_id]  
            fused_results.append(result)  
        return fused_results  
  
    # --- チャット履歴の表示 ---  
    for message in st.session_state["main_chat_messages"]:  
        with st.chat_message(message["role"]):  
            st.markdown(message["content"])  
  
    # --- 検索クエリの表示（チャット履歴の直後） ---  
    if st.session_state.get("last_search_query"):  
        st.markdown(  
            f"""<div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-bottom: 10px;">  
            <b>🔍 使用された検索クエリ：</b><br>  
            <code>{st.session_state['last_search_query']}</code>  
            </div>  
            """, unsafe_allow_html=True  
        )  
  
    # --- チャット入力欄 ---  
    prompt = st.chat_input("ご質問を入力してください:")  
  
    # --- 入力受付 ---  
    if prompt:  
        user_input = prompt  
        recent_messages = [m["content"] for m in st.session_state["main_chat_messages"]][-4:]  
  
        if st.session_state.get("query_generation_method", "シンプル結合") == "クエリリライト":  
            with st.spinner("検索用クエリを生成中..."):  
                search_query = rewrite_query(user_input, recent_messages, st.session_state["system_message"])  
        else:  
            search_query = user_input + " " + " ".join(recent_messages)  
  
        # 検索クエリをセッションに保存し、即座に再描画  
        st.session_state["last_search_query"] = search_query  
        st.session_state["main_chat_messages"].append({"role": "user", "content": user_input})  
        st.rerun()  # ここで再描画  
  
    # --- ユーザー入力後の処理 ---  
    if (  
        st.session_state.get("last_search_query")  
        and st.session_state["main_chat_messages"]  
        and st.session_state["main_chat_messages"][-1]["role"] == "user"  
        and (not st.session_state["main_chat_messages"][-1].get("handled"))  
    ):  
        search_query = st.session_state["last_search_query"]  
        user_input = st.session_state["main_chat_messages"][-1]["content"]  
  
        if search_mode == "セマンティック検索":  
            search_results = keyword_semantic_search(search_query, topNDocuments=topNDocuments, strictness=strictness)  
        elif search_mode == "ベクトル検索":  
            search_results = keyword_vector_search(search_query, topNDocuments=topNDocuments)  
        elif search_mode == "ハイブリッド検索":  
            search_results = hybrid_search(search_query, topNDocuments=topNDocuments, strictness=strictness)  
  
        context_parts = []  
        for result in search_results:  
            title = result.get("title", "タイトルなし")  
            content_val = result.get("content", "")  
            context_parts.append(f"ファイル名: {title}\n内容: {content_val}")  
        context = "\n".join(context_parts)  
        initial_context = f"以下のコンテキストを参考にしてください: {context[:50000]}"  
  
        rule_message = (  
            "回答する際は、以下のルールに従ってください：\n"  
            "必要に応じて、提供されたコンテキストを参照してください。\n"  
        )  
  
        past_message_count = st.session_state.get("past_message_count", 10)  
        num_messages_to_include = past_message_count * 2  
        messages = []  
        messages.append({"role": "system", "content": st.session_state["system_message"]})  
        messages.append({"role": "user", "content": rule_message})  
  
        initial_context_message = f"以下のコンテキストを参考にしてください: {context[:40000]}"  
        if st.session_state["images"]:  
            image_contents = [  
                {  
                    "type": "image_url",  
                    "image_url": {"url": f"data:image/jpeg;base64,{img['encoded']}"}  
                }  
                for img in st.session_state["images"]  
            ]  
            messages.append({"role": "user", "content": [{"type": "text", "text": initial_context_message}] + image_contents})  
        else:  
            messages.append({"role": "user", "content": initial_context_message})  
  
        messages.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state["main_chat_messages"][-(num_messages_to_include):]])  
  
        try:  
            call_args = dict(  
                model=model_to_use,  
                messages=messages,  
            )  
            if model_to_use in ["o1", "o4-mini", "o3"]:  
                call_args["reasoning_effort"] = st.session_state.get("reasoning_effort", "medium")  
  
            response = client.chat.completions.create(**call_args)  
            assistant_response = response.choices[0].message.content  
  
            st.session_state["main_chat_messages"].append({"role": "assistant", "content": assistant_response})  
            st.session_state["main_chat_messages"][-2]["handled"] = True  # この質問は処理済み  
  
            with st.chat_message("assistant"):  
                st.markdown(assistant_response)  
  
            st.write("### 引用文：")  
            if search_results:  
                for i, result in enumerate(search_results):  
                    score = result.get("fusion_score", result.get("@search.score", 0))  
                    filepath = result.get("filepath")  
                    url = result.get("url")  
                    if filepath:  
                        file_name = os.path.basename(filepath)  
                    else:  
                        file_name = "ファイル名不明"  
                    with st.expander(f"ドキュメント {i+1} - {file_name}（スコア: {score:.2f}）", expanded=False):  
                        st.caption(result.get("content", ""))  
                        if url:  
                            sas_url = generate_sas_url(url, file_name)  
                            st.markdown(f"[{file_name}]({sas_url}) をダウンロードするにはここをクリックしてください。")  
                        else:  
                            st.write("このファイルはダウンロードできません。")  
            else:  
                st.write("引用なし")  
  
            current_index = st.session_state.get("current_chat_index")  
            if current_index is not None:  
                st.session_state.sidebar_messages[current_index]["messages"] = st.session_state["main_chat_messages"].copy()  
                if not st.session_state.sidebar_messages[current_index].get("first_assistant_message"):  
                    st.session_state.sidebar_messages[current_index]["first_assistant_message"] = assistant_response  
  
                save_chat_history()  
            st.session_state["images"] = []  
  
        except Exception as e:  
            st.error(f"エラーが発生しました: {e}")  
  
    current_index = st.session_state.get("current_chat_index")  
    if current_index is not None:  
        st.session_state.sidebar_messages[current_index]["messages"] = st.session_state["main_chat_messages"].copy()  
        st.session_state.sidebar_messages[current_index]["system_message"] = st.session_state["system_message"]  
        save_chat_history()  
  
if __name__ == "__main__":  
    main()  