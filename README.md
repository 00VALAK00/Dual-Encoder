## Dual Encoder as a retriever for Large Language Models 
 
### 1. Vanilla RAG
In a basic RAG application the two most important components are the LLM and the retriever.<p>
The retriever provided the prompt, looks for the k most relevant chunks retrieves and add them to the llm's context.
The llm takes the prompt, the added context and starts inference.

### 2. The retriever
The retriever plays a major role that affects the output of the llm. A bad retriever will lead to bad responses and even hullucinations. Thus the selection or conception of a good retriever model is a must before even thinking about doing RAG applications. You can find in this repo the implementation of a DualEncoder architecture for Q&A and everything you need to build your retriever model.

### 3. About the retriever 
The DualEncoder uses 2 bert-based encoders one for the prompt (question) the other for the answer(s) or the chunks of text in case of retrieval.
The dual encoder uses a neat trick leveraging the cross entropy loss to assign better scores for the most relevant answers. This way the model learns to select the best answer(s) for a given question.

