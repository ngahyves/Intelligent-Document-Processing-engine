from src.rag.chunker import DocumentChunker
from src.rag.embedder import TextEmbedder

# 1. Simuler un texte OCR (prends celui que ton OCR a généré !)
text_example = "From Original Massa9asvy Carclyn Sont Yeanesday February 232000 1249 PM" \
" Ryzn Thcmas Desel Paula Chanitan Gavalti Daragar Karen Chalkin, Karcn Prcil ," \
" Michacl WcComick, Brendan Canovale Nary _ Subject RE YAP Mossagos" \
" Im fine with these points pls get with KD and KC regarding our desire" \
" to also havo esponse to queshion Iike dont you encourage parents who smcke smoking," \
" rather than keep an eye on their cigarettes thanks 70fvn Ecr Youl Smking Freveniem h 223 why quit"

# 2. Chunking
chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
chunks = chunker.split_text(text_example)
print(f"Chunks : {chunks}")

# 3. Embedding
embedder = TextEmbedder()
vectors = embedder.embed_chunks(chunks)
print(f"Form of first vector : {vectors[0].shape}") 
# Doit afficher (384,)