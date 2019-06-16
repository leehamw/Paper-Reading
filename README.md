# Paper-Reading
Paper reading list in natural language processing.


- [Paper-Reading](#paper-reading)
  - [Deep Learning in NLP](#deep-learning-in-nlp)
  - [Dialogue System](#dialogue-system)
  - [Text Summarization](#text-summarization)
  - [Topic Modeling](#topic-modeling)
  - [Machine Translation](#machine-translation)
  - [Question Answering](#question-answering)
  - [Reading Comprehension](#reading-comprehension)
  - [Image Captioning](#image-captioning)

***

## Deep Learning in NLP
* **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". NAACL(2019) [[PDF]](./papers/deep-learning/N19-1423.pdf) [[code]](https://github.com/google-research/bert)
* **CNM**: "CNM: An Interpretable Complex-valued Network for Matching". NAACL(2019) [[PDF]](./papers/deep-learning/N19-1420.pdf) [[code]](https://github.com/wabyking/qnn)
* **ELMo**: "Deep contextualized word representations". NAACL(2018) [[PDF]](./papers/deep-learning/N18-1202.pdf)
* **Survey on Attention**: "An Introductory Survey on Attention Mechanisms in NLP Problems". arXiv(2018) [[PDF]](./papers/deep-learning/1811.05544.pdf)
* **VAE**: "An Introduction to Variational Autoencoders". arXiv(2019) [[PDF]](./papers/deep-learning/1906.02691.pdf)
* **Transformer**: "Attention is All you Need". NIPS(2017) [[PDF]](./papers/deep-learning/7181-attention-is-all-you-need.pdf) [[code-official]](https://github.com/tensorflow/tensor2tensor) [[code-tf]](https://github.com/Kyubyong/transformer) [[code-py]](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
* **Transformer-XL**: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context". arXiv(2019) [[PDF]](./papers/deep-learning/1901.02860.pdf) [[code]](https://github.com/kimiyoung/transformer-xl)
* **ConvS2S**: "Convolutional Sequence to Sequence Learning". ICML(2017) [[PDF]](./papers/deep-learning/gehring17a-convs2s.pdf)
* **Additive Attention**: "Neural Machine Translation by Jointly Learning to Align and Translate". ICLR(2015) [[PDF]](./papers/deep-learning/1409.0473-nmt-align.pdf) 
* **Multiplicative Attention**: "Effective Approaches to Attention-based Neural Machine Translation". EMNLP(2015) [[PDF]](./papers/deep-learning/D15-1166.pdf)
* **Memory Net**: "End-To-End Memory Networks". NIPS(2015) [[PDF]](./papers/deep-learning/5846-end-to-end-memory-networks.pdf)
* **Pointer Net**: "Pointer Networks". NIPS(2015) [[PDF]](./papers/deep-learning/5866-pointer-networks.pdf) 
* **Copying Mechanism**: "Incorporating Copying Mechanism in Sequence-to-Sequence Learning". ACL(2016) [[PDF]](./papers/deep-learning/P16-1154.pdf)
* **Coverage Mechanism**: "Modeling Coverage for Neural Machine Translation". ACL(2016) [[PDF]](./papers/deep-learning/P16-1008.pdf)
* **GAN**: "Generative Adversarial Nets". NIPS(2014) [[PDF]](./papers/deep-learning/5423-generative-adversarial-nets.pdf)
* **SeqGAN**: "SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient". AAAI(2017) [[PDF]](./papers/deep-learning/14344-66977-1-PB.pdf) [[code]](https://github.com/LantaoYu/SeqGAN)
* **MacNet**: "MacNet: Transferring Knowledge from Machine
Comprehension to Sequence-to-Sequence Models". NIPS(2018) [[PDF]](./papers/text-summarization/7848-macnet.pdf)
* **Graph2Seq**: "Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks". arXiv(2018) [[PDF]](./papers/deep-learning/1804.00823-graph2seq.pdf)
* **Pretrained Seq2Seq**: "Unsupervised Pretraining for Sequence to Sequence Learning". EMNLP(2017) [[PDF]](./papers/deep-learning/D17-1039.pdf)
* **Multi-task Learning**: "An Overview of Multi-Task Learning in Deep Neural Networks". arXiv(2017) [[PDF]](./papers/deep-learning/1706.05098.pdf)
* **Latent Multi-task**: "Latent Multi-task Architecture Learning". AAAI(2019) [[PDF]](./papers/deep-learning/1705.08142.pdf) [[code]](https://github.com/sebastianruder/sluice-networks)


## Dialogue System
* **TRADE**: "Transferable Multi-Domain State Generator for Task-Oriented
Dialogue Systems". ACL(2019) [[PDF]](./papers/dialogue-system/1905.08743.pdf) [[code]](https://github.com/jasonwu0731/trade-dst) :star::star::star::star:
* **PAML**: "Personalizing Dialogue Agents via Meta-Learning". ACL(2019) [[PDF]](./papers/dialogue-system/1905.10033.pdf) [[code]](https://github.com/HLTCHKUST/PAML) :star::star::star:
* **Pretrain-Fine-tune**: "Training Neural Response Selection for Task-Oriented Dialogue Systems". ACL(2019) [[PDF]](./papers/dialogue-system/1906.01543.pdf) [[data]](https://github.com/PolyAI-LDN/conversational-datasets) :star::star::star:
* **PostKS**: "Learning to Select Knowledge for Response Generation in Dialog Systems". arXiv(2019) [[PDF]](./papers/dialogue-system/1902.04911.pdf) :star::star:
* **GLMP**: "Global-to-local Memory Pointer Networks for Task-Oriented Dialogue". ICLR(2019) [[PDF]](./papers/dialogue-system/1901.04713.pdf) [[code]](https://github.com/jasonwu0731/GLMP) :star::star::star::star:
* **Two-Stage-Transformer**: "Wizard of Wikipedia: Knowledge-Powered Conversational agents". ICLR(2019) [[PDF]](./papers/dialogue-system/1811.01241.pdf) :star::star:
* **Multi-level Mem**: "Multi-Level Memory for Task Oriented Dialogs". NAACL(2019) [[PDF]](./papers/dialogue-system/N19-1375.pdf)  :star::star::star:
* **BossNet**: "Disentangling Language and Knowledge in Task-Oriented Dialogs
". NAACL(2019) [[PDF]](./papers/dialogue-system/N19-1126.pdf) [[code]](https://github.com/dair-iitd/BossNet) :star::star::star:
* **CAS**: "Skeleton-to-Response: Dialogue Generation Guided by Retrieval Memory". NAACL(2019) [[PDF]](./papers/dialogue-system/N19-1124.pdf) [[code]](https://github.com/jcyk/Skeleton-to-Response) :star::star::star:
* **Edit-N-Rerank**: "Response Generation by Context-aware Prototype Editing". AAAI(2019) [[PDF]](./papers/dialogue-system/1806.07042.pdf) [[code]](https://github.com/MarkWuNLP/ResponseEdit) :star::star::star:
* **Survey of Dialogue Corpora**: "A Survey of Available Corpora For Building Data-Driven Dialogue Systems: The Journal Version". Dialogue & Discourse(2018) [[PDF]](./papers/dialogue-system/3690-7705-1-PB.pdf) :star:
* **D2A**: "Dialog-to-Action: Conversational Question Answering Over a Large-Scale Knowledge Base". NIPS(2018) [[PDF]](./papers/dialogue-system/7558-dialog-to-action.pdf) :star::star:
* **DAIM**: "Generating Informative and Diverse Conversational Responses via Adversarial Information Maximization". NIPS(2018) [[PDF]](./papers/dialogue-system/7452-generating-informative.pdf) :star::star:
* **LU-DST**: "Multi-task Learning for Joint Language Understanding and Dialogue State Tracking". SIGDIAL(2018) [[PDF]](./papers/dialogue-system/W18-5045.pdf)  :star::star::star:
* **MTask**: "A Knowledge-Grounded Neural Conversation Model". AAAI(2018)  [[PDF]](./papers/dialogue-system/16710-76819-1-PB.pdf) :star:
* **MTask-M**: "Multi-Task Learning for Speaker-Role Adaptation in Neural Conversation Models". IJCNLP(2018) [[PDF]](./papers/dialogue-system/I17-1061.pdf) :star:
* **GenDS**: "Flexible End-to-End Dialogue System for Knowledge Grounded Conversation". arXiv(2017) [[PDF]](./papers/dialogue-system/1709.04264.pdf) :star::star:
* **SL+RL**: "Dialogue Learning with Human Teaching and Feedback in End-to-End Trainable Task-Oriented Dialogue Systems". NAACL(2018) [[PDF]](./papers/dialogue-system/N18-1187.pdf) :star::star::star:
* **Time-Decay-SLU**: "How Time Matters: Learning Time-Decay Attention for Contextual Spoken Language Understanding in Dialogues". NAACL(2018) [[PDF]](./papers/dialogue-system/N18-1194.pdf) [[code]](https://github.com/MiuLab/Time-Decay-SLU) :star::star::star::star:
* **REASON**: "Dialog Generation Using Multi-turn Reasoning Neural Networks". NAACL(2018) [[PDF]](./papers/dialogue-system/N18-1186.pdf) :star::star::star:
* **ADVMT**: "One “Ruler” for All Languages: Multi-Lingual Dialogue Evaluation with Adversarial Multi-Task Learning". IJCAI(2018) [[PDF]](./papers/dialogue-system/IJCAI-0616.pdf) :star::star:
* **STD/HTD**: "Learning to Ask Questions in Open-domain Conversational Systems with Typed Decoders". ACL(2018) [[PDF]](./papers/dialogue-system/P18-1204.pdf) [[code]](https://github.com/victorywys/Learning2Ask_TypedDecoder) :star::star::star:
* **CSF used**: "Generating Informative Responses with Controlled Sentence Function". ACL(2018) [[PDF]](./papers/dialogue-system/P18-1139.pdf) [[code]](https://github.com/kepei1106/SentenceFunction) :star::star::star:
* **Mem2Seq**: "Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems". ACL(2018) [[PDF]](./papers/dialogue-system/P18-1136.pdf) [[code]](https://github.com/HLTCHKUST/Mem2Seq) :star::star::star::star:
* **NKD**: "Knowledge Diffusion for Neural Dialogue Generation". ACL(2018) [[PDF]](./papers/dialogue-system/P18-1138.pdf) [[data]](https://github.com/liushuman/neural-knowledge-diffusion) :star::star:
* **DAWnet**: "Chat More: Deepening and Widening the Chatting Topic via A Deep Model". SIGIR(2018) [[PDF]](./papers/dialogue-system/p255-wang.pdf) [[code]](https://sigirdawnet.wixsite.com/dawnet) :star::star::star:
* **ZSDG**: "Zero-Shot Dialog Generation with Cross-Domain Latent Actions". SIGDIAL(2018) [[PDF]](./papers/dialogue-system/W18-5001.pdf) [[code]](https://github.com/snakeztc/NeuralDialog-ZSDG) :star::star::star:
* **DUA**: "Modeling Multi-turn Conversation with Deep Utterance Aggregation". COLING(2018) [[PDF]](./papers/dialogue-system/C18-1317.pdf) [[code]](https://github.com/cooelf/DeepUtteranceAggregation) :star::star:
* **Data-Aug**: "Sequence-to-Sequence Data Augmentation for Dialogue Language Understanding". COLING(2018) [[PDF]](./papers/dialogue-system/C18-1105.pdf) [[code]](https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU) :star::star:
* **DSR**: "Sequence-to-Sequence Learning for Task-oriented Dialogue with Dialogue State Representation". COLING(2018)  [[PDF]](./papers/dialogue-system/C18-1320.pdf) :star::star:
* **DC-MMI**: "Generating More Interesting Responses in Neural Conversation Models with Distributional Constraints". EMNLP(2018) [[PDF]](./papers/dialogue-system/D18-1431.pdf) [[code]](https://github.com/abaheti95/DC-NeuralConversation) :star::star:
* **StateNet**: "Towards Universal Dialogue State Tracking". EMNLP(2018) [[PDF]](./papers/dialogue-system/D18-1299.pdf) :star:
* **cVAE-XGate/CGate**: "Better Conversations by Modeling, Filtering, and Optimizing for Coherence and Diversity". EMNLP(2018) [[PDF]](./papers/dialogue-system/D18-1432.pdf) [[code]](https://github.com/XinnuoXu/CVAE_Dial) :star::star::star:
* **SMN**: "Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots". ACL(2017) [[PDF]](./papers/dialogue-system/P17-1046.pdf)  [[code]](https://github.com/MarkWuNLP/MultiTurnResponseSelection) :star::star::star::star:
* **KVR Net**: "Key-Value Retrieval Networks for Task-Oriented Dialogue". SIGDIAL(2017) [[PDF]](./papers/dialogue-system/W17-5506.pdf) [[data]](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/) :star::star:
* **MMI**: "A Diversity-Promoting Objective Function for Neural Conversation Models". NAACL-HLT(2016)  [[PDF]](./papers/dialogue-system/N16-1014.pdf) [[code]](https://github.com/jiweil/Neural-Dialogue-Generation) :star::star:
* **RL-Dialogue**: "Deep Reinforcement Learning for Dialogue Generation". EMNLP(2016) [[PDF]](./papers/dialogue-system/D16-1127.pdf) :star:
* **TA-Seq2Seq**: "Topic Aware Neural Response Generation". AAAI(2017) [[PDF]](./papers/dialogue-system/AAAI17_TA-Seq2Seq.pdf) [[code]](https://github.com/LynetteXing1991/TA-Seq2Seq) :star::star:
* **MA**: "Mechanism-Aware Neural Machine for Dialogue Response Generation". AAAI(2017) [[PDF]](./papers/dialogue-system/14471-66751-1-PB.pdf) :star::star:
* **HRED**: "Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models". AAAI(2016) [[PDF]](./papers/dialogue-system/11957-56353-1-PB.pdf) [[code]](https://github.com/julianser/hed-dlg) :star::star:
* **VHRED**: "A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues". AAAI(2017) [[PDF]](./papers/dialogue-system/14567-66703-1-PB.pdf) [[code]](https://github.com/julianser/hed-dlg-truncated) :star::star:
* **CVAE/KgCVAE**: "Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders". ACL(2017) [[PDF]](./papers/dialogue-system/P17-1061.pdf) [[code]](https://github.com/snakeztc/NeuralDialog-CVAE) :star::star::star:
* **ERM**: "Elastic Responding Machine for Dialog Generation with Dynamically Mechanism Selecting". AAAI(2018) [[PDF]](./papers/dialogue-system/16316-76896-1-PB.pdf) :star::star:
* **Tri-LSTM**: "Augmenting End-to-End Dialogue Systems With Commonsense Knowledge". AAAI(2018) [[PDF]](./papers/dialogue-system/16573-76790-1-PB.pdf) :star::star:
* **Dual Fusion**: "Smarter Response with Proactive Suggestion: A New Generative Neural Conversation Paradigm". IJCAI(2018) [[PDF]](./papers/dialogue-system/IJCAI-0629.pdf) :star::star::star:
* **CCM**: "Commonsense Knowledge Aware Conversation Generation with Graph Attention". IJCAI(2018) [[PDF]](./papers/dialogue-system/IJCAI-0643.pdf) [[code]](https://github.com/tuxchow/ccm) :star::star::star::star::star:
* **PCCM**: "Assigning Personality/Profile to a Chatting Machine for Coherent Conversation Generation". IJCAI(2018) [[PDF]](./papers/dialogue-system/IJCAI-0595.pdf) [[code]](https://github.com/qianqiao/AssignPersonality) :star::star::star::star:
* **ECM**: "Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory". AAAI(2018) [[PDF]](./papers/dialogue-system/16455-76513-1-PB.pdf) [[code]](https://github.com/tuxchow/ecm) :star::star::star::star:
* **Topic-Seg-Label**: "A Weakly Supervised Method for Topic Segmentation and Labeling in Goal-oriented Dialogues via Reinforcement Learning". IJCAI(2018) [[PDF]](./papers/dialogue-system/IJCAI-0612.pdf) [[code]](https://github.com/truthless11/Topic-Seg-Label) :star::star::star::star:
* **AliMe**: "AliMe Chat: A Sequence to Sequence and Rerank based Chatbot Engine". ACL(2017) [[PDF]](./papers/dialogue-system/P17-2079.pdf) :star:
* **Retrieval+multi-seq2seq**: "An Ensemble of Retrieval-Based and Generation-Based Human-Computer Conversation Systems". IJCAI(2018) [[PDF]](./papers/dialogue-system/IJCAI-0609.pdf) :star::star::star:

## Text Summarization
* **BERT-Two-Stage**: "Pretraining-Based Natural Language Generation for Text Summarization". arXiv(2019)  [[PDF]](./papers/text-summarization/1902.09243.pdf) :star::star:
* **QASumm**: "Guiding Extractive Summarization with Question-Answering Rewards". NAACL(2019) [[PDF]](./papers/text-summarization/N19-1264.pdf) [[code]](https://github.com/ucfnlp/summ_qa_rewards) :star::star::star::star:
* **Re^3Sum**: "Retrieve, Rerank and Rewrite: Soft Template Based Neural Summarization". ACL(2018) [[PDF]](./papers/text-summarization/P18-1015.pdf) [[code]](http://www4.comp.polyu.edu.hk/~cszqcao/data/IRSum_Resource.zip) :star::star::star:
* **NeuSum**: "Neural Document Summarization by Jointly Learning to Score and Select Sentences". ACL(2018) [[PDF]](./papers/text-summarization/P18-1061.pdf) :star::star::star:
* **rnn-ext+abs+RL+rerank**: "Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting". ACL(2018) [[PDF]](./papers/text-summarization/P18-1063.pdf) [[code]](https://github.com/ChenRocks/fast_abs_rl) :star::star::star::star::star:
* **Seq2Seq+CGU**: "Global Encoding for Abstractive Summarization". ACL(2018) [[PDF]](./papers/text-summarization/P18-2027.pdf) [[code]](https://github.com/lancopku/Global-Encoding) :star::star::star:
* **T-ConvS2S**: "Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization". EMNLP(2018) [[PDF]](./papers/text-summarization/D18-1206.pdf) [[code]](https://github.com/shashiongithub/XSum) :star::star::star::star:
* **RL-Topic-ConvS2S**: "A reinforced topic-aware convolutional sequence-to-sequence model for abstractive text summarization." IJCAI (2018) [[PDF]](./papers/text-summarization/0619.pdf) :star::star::star:
* **GANsum**: "Generative Adversarial Network for Abstractive Text Summarization". AAAI (2018) [[PDF]](./papers/text-summarization/16238-77257-1-PB.pdf) :star:
* **FTSum**: "Faithful to the Original: Fact Aware Neural Abstractive Summarization". AAAI(2018) [[PDF]](./papers/text-summarization/16121-76767-1-PB.pdf) :star::star:
* **PGC**: "Get To The Point: Summarization with Pointer-Generator Networks". ACL (2017) [[PDF]](./papers/text-summarization/P17-1099.pdf) [[code]](https://github.com/abisee/pointer-generator) :star::star::star::star::star:
* **ABS/ABS+**: "A Neural Attention Model for Abstractive Sentence Summarization". EMNLP (2015) [[PDF]](./papers/text-summarization/D15-1044.pdf) :star::star:
* **RAS-Elman/RAS-LSTM**: "Abstractive Sentence Summarization with Attentive Recurrent Neural Networks. NAACL (2016) [[PDF]](./papers/text-summarization/N16-1012.pdf) [[code]](https://github.com/facebookarchive/NAMAS)  :star::star::star:
* **words-lvt2k-1sent**: "Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond". CoNLL (2016) [[PDF]](./papers/text-summarization/K16-1028.pdf) :star:

## Topic Modeling
* **LDA**: "Latent Dirichlet Allocation". JMLR (2003) [[PDF]](./papers/topic-modeling/JMLR03_LDA_BleiNgJordan.pdf) [[code]](https://github.com/blei-lab/lda-c) :star::star::star::star::star:
* **Parameter Estimation**: "Parameter estimation for text analysis." Technical report (2005). [[PDF]](./papers/topic-modeling/Parameter_estimation_for_text_analysis.pdf) :star::star::star:
* **DTM**: "Dynamic Topic Models". ICML (2006) [[PDF]](./papers/topic-modeling/ICML06_DTM.pdf) [[code]](https://github.com/blei-lab/dtm) :star::star::star::star:
* **cDTM**: "Continuous Time Dynamic Topic Models". arXiv (2012) [[PDF]](./papers/topic-modeling/arXiv12_cDTM.pdf) :star::star:
* **NTM**: "A Novel Neural Topic Model and Its Supervised Extension". AAAI (2015) [[PDF]](./papers/topic-modeling/AAAI15_NTM.pdf) :star::star::star::star:
* **TWE**: "Topical Word Embeddings". AAAI (2015) [[PDF]](./papers/topic-modeling/AAAI15_TWE.pdf) :star::star:
* **RATM-D**: Recurrent Attentional Topic Model. AAAI (2017)[[PDF]](./papers/topic-modeling/AAAI17_RATM-D.pdf) :star::star::star::star:
* **RIBS-TM**: "Don't Forget the Quantifiable Relationship between Words: Using Recurrent Neural Network for Short Text Topic Discovery". AAAI (2017) [[PDF]](./papers/topic-modeling/AAAI17_RIBS_TM.pdf) :star::star::star:
* **Topic coherence**: "Optimizing Semantic Coherence in Topic Models". EMNLP (2011) [[PDF]](./papers/topic-modeling/EMNLP11_Topic_Coherence.pdf) :star::star::star:
* **Topic coherence**: "Automatic Evaluation of Topic Coherence". NAACL (2010) [[PDF]](./papers/topic-modeling/Automatic_Evaluation_of_Topic_Coherence.pdf) :star::star::star:
* **DADT**: "Authorship Attribution with Author-aware Topic Models". ACL(2012) [[PDF]](./papers/topic-modeling/P12-2052.pdf) :star::star::star::star:
* **Gaussian-LDA**: "Gaussian LDA for Topic Models with Word Embeddings". ACL (2015) [[PDF]](./papers/topic-modeling/ACL15_Gaussian_LDA.pdf) [[code]](https://github.com/rajarshd/Gaussian_LDA) :star::star::star::star:
* **LFTM**:	"Improving Topic Models with Latent Feature Word Representations". TACL (2015) [[PDF]](./papers/topic-modeling/TACL15_LFTM.pdf) [[code]](https://github.com/datquocnguyen/LFTM) :star::star::star::star::star:
* **TopicVec**: "Generative Topic Embedding: a Continuous Representation of Documents". ACL (2016) [[PDF]](./papers/topic-modeling/ACL16_TopicVec.pdf) [[code]](https://github.com/askerlee/topicvec) :star::star::star::star:
* **SLRTM**: "Sentence Level Recurrent Topic Model: Letting Topics Speak for Themselves". arXiv (2016) [[PDF]](./papers/topic-modeling/arXiv16_SLRTM.pdf) :star::star:
* **TopicRNN**: "TopicRNN: A Recurrent Neural Network with Long-Range Semantic Dependency". ICLR(2017) [[PDF]](./papers/topic-modeling/ICLR17_TopicRNN.pdf) [[code]](https://github.com/dangitstam/topic-rnn) :star::star::star::star::star:
* **NMF boosted**: "Stability of topic modeling via matrix factorization". Expert Syst. Appl. (2018) [[PDF]](./papers/topic-modeling/ESA18_Stability_of_topic_modeling_via_matrix_factorization.pdf) :star::star:
* **Evaluation of Topic Models**: "External Evaluation of Topic Models". Australasian Doc. Comp. Symp. (2009) [[PDF]](./papers/topic-modeling/External_Evaluation_of_Topic_Models.pdf) :star::star:
* **Topic2Vec**: "Topic2Vec: Learning distributed representations of topics". IALP (2015) [[PDF]](./papers/topic-modeling/IALP15_Topic2Vec.pdf) :star::star::star:
* **L-EnsNMF**: "L-EnsNMF: Boosted Local Topic Discovery via Ensemble of Nonnegative Matrix Factorization". ICDM (2016) [[PDF]](./papers/topic-modeling/ICDM16_L-EnsNMF.pdf) [[code]](https://github.com/sanghosuh/lens_nmf-matlab) :star::star::star::star::star:
* **DC-NMF**: "DC-NMF: nonnegative matrix factorization based on divide-and-conquer for fast clustering and topic modeling". J. Global Optimization (2017) [[PDF]](./papers/topic-modeling/JGO17_DC_NMF.pdf) :star::star::star:
* **cFTM**: "The contextual focused topic model". KDD (2012) [[PDF]](./papers/topic-modeling/KDD12_cFTM.pdf) :star::star::star:
* **CLM**: "Collaboratively Improving Topic Discovery and Word Embeddings by Coordinating Global and Local Contexts". KDD (2017) [[PDF]](./papers/topic-modeling/KDD17_CLM.pdf) [[code]](https://github.com/XunGuangxu/2in1) :star::star::star::star::star:
* **GMTM**: "Unsupervised Topic Modeling for Short Texts Using Distributed Representations of Words". NAACL (2015) [[PDF]](./papers/topic-modeling/NAACL15_GMTM.pdf) :star::star::star::star:
* **GPU-PDMM**: "Enhancing Topic Modeling for Short Texts with Auxiliary Word Embeddings". TOIS (2017) [[PDF]](./papers/topic-modeling/TOIS17_GPU-PDMM.pdf) :star::star::star:
* **BPT**: "A Two-Level Topic Model Towards Knowledge Discovery from Citation Networks". TKDE (2014) [[PDF]](./papers/topic-modeling/TKDE14_BPT.pdf) :star::star::star:
* **BTM**: "A Biterm Topic Model for Short Texts". WWW (2013) [[PDF]](./papers/topic-modeling/WWW13_BTM.pdf) [[code]](https://github.com/xiaohuiyan/BTM) :star::star::star::star:
* **HGTM**: "Using Hashtag Graph-Based Topic Model to Connect Semantically-Related Words Without Co-Occurrence in Microblogs". TKDE (2016) [[PDF]](./papers/topic-modeling/TKDE16_HGTM.pdf) :star::star::star:
* **COTM**: "A topic model for co-occurring normal documents and short texts". WWW (2018) [[PDF]](./papers/topic-modeling/WWW18_COTM.pdf) :star::star::star::star:

## Machine Translation
* **Deliberation Networks**: "Deliberation Networks: Sequence Generation Beyond One-Pass Decoding". NIPS(2017) [[PDF]](./papers/machine-translation/6775-deliberation-networks.pdf) :star::star::star:
* **Multi-pass decoder**: "Adaptive Multi-pass Decoder for Neural Machine Translation". EMNLP(2018) [[PDF]](./papers/machine-translation/Multi-pass-decoder.pdf) :star::star::star:
* **KVMem-Attention**: "Neural Machine Translation with Key-Value Memory-Augmented Attention". IJCAI(2018) [[PDF]](./papers/machine-translation/IJCAI-0357.pdf) :star::star::star:

## Question Answering
* **CFC**: "Coarse-grain Fine-grain Coattention Network for Multi-evidence Question Answering". ICLR(2019) [[PDF]](./papers/question-answering/1901.00603.pdf) :star::star:
* **MTQA**: "Multi-Task Learning with Multi-View Attention for Answer Selection and Knowledge Base Question Answering". AAAI(2019) [[PDF]](./papers/question-answering/1812.02354.pdf) [[code]](https://github.com/dengyang17/MTQA) :star::star::star:
* **CQG-KBQA**: "Knowledge Base Question Answering via Encoding of
Complex Query Graphs". EMNLP(2018) [[PDF]](./papers/question-answering/D18-1242.pdf) [[code]](http://202.120.38.146/CompQA/) :star::star::star::star::star:
* **HR-BiLSTM**: "Improved Neural Relation Detection for Knowledge Base Question Answering". ACL(2017) [[PDF]](./papers/question-answering/P17-1053.pdf) :star::star::star:
* **KBQA-CGK**: "An End-to-End Model for Question Answering over Knowledge Base with Cross-Attention Combining Global Knowledge". ACL(2017) [[PDF]](./papers/question-answering/P17-1021.pdf) :star::star::star:
* **KVMem**: "Key-Value Memory Networks for Directly Reading Documents". EMNLP(2016) [[PDF]](./papers/question-answering/D16-1147.pdf) :star::star::star:

## Reading Comprehension
* **DecompRC**: "Multi-hop Reading Comprehension through Question Decomposition and Rescoring". ACL(2019) [[PDF]](./papers/reading-comprehension/1906.02916.pdf) [[code]](https://github.com/shmsw25/DecompRC) :star::star::star::star:
* **FlowQA**: "FlowQA: Grasping Flow in History for Conversational Machine Comprehension". ICLR(2019) [[PDF]](./papers/reading-comprehension/1810.06683.pdf) [[code]](https://github.com/momohuang/FlowQA) :star::star::star::star::star:
*  **SDNet**: "SDNet: Contextualized Attention-based Deep Network for Conversational Question Answering". arXiv(2018) [[PDF]](./papers/reading-comprehension/1812.03593.pdf) [[code]](https://github.com/microsoft/SDNet) :star::star::star::star:

## Image Captioning
* **MLAIC**: "A Multi-task Learning Approach for Image Captioning". IJCAI(2018) [[PDF]](./papers/image-captioning/2018-ijcai-multitask-final.pdf) [[code]](https://github.com/andyweizhao/Multitask_Image_Captioning) :star::star::star::star:
* **Up-Down Attention**: "Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering". CVPR(2018) [[PDF]](./papers/image-captioning/Bottom-Up_and_Top-Down_CVPR_2018_paper.pdf) :star::star::star:
* **Recurrent-RSA**: "Pragmatically Informative Image Captioning with Character-Level Inference". NAACL(2018) [[PDF]](./papers/image-captioning/NAACL18-incremental-RSA.pdf) [[code]](https://github.com/reubenharry/Recurrent-RSA) :star::star::star: