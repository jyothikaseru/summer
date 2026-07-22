# dataset.py

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from common.config import PAD_IDX

# Toy English-French sentence pairs

sentence_pairs = [
    ("i love ai", "j'aime l'ia"),
    ("i love machine learning", "j'aime l'apprentissage automatique"),
    ("i study artificial intelligence", "j'étudie l'intelligence artificielle"),
    ("i learn deep learning", "j'apprends l'apprentissage profond"),
    ("i use python", "j'utilise python"),
    ("i write code", "j'écris du code"),
    ("i read books", "je lis des livres"),
    ("i drink coffee", "je bois du café"),
    ("i eat apples", "je mange des pommes"),
    ("i like music", "j'aime la musique"),

    ("you love ai", "tu aimes l'ia"),
    ("you study mathematics", "tu étudies les mathématiques"),
    ("you play football", "tu joues au football"),
    ("you drink tea", "tu bois du thé"),
    ("you eat bananas", "tu manges des bananes"),
    ("you read newspapers", "tu lis des journaux"),
    ("you watch movies", "tu regardes des films"),
    ("you build robots", "tu construis des robots"),
    ("you create projects", "tu crées des projets"),
    ("you learn python", "tu apprends python"),

    ("he likes coffee", "il aime le café"),
    ("he studies physics", "il étudie la physique"),
    ("he plays basketball", "il joue au basket"),
    ("he reads books", "il lit des livres"),
    ("he writes letters", "il écrit des lettres"),
    ("he buys a computer", "il achète un ordinateur"),
    ("he drives a car", "il conduit une voiture"),
    ("he watches television", "il regarde la télévision"),
    ("he learns numpy", "il apprend numpy"),
    ("he visits paris", "il visite paris"),

    ("she likes tea", "elle aime le thé"),
    ("she studies biology", "elle étudie la biologie"),
    ("she plays tennis", "elle joue au tennis"),
    ("she reads novels", "elle lit des romans"),
    ("she writes poems", "elle écrit des poèmes"),
    ("she buys flowers", "elle achète des fleurs"),
    ("she drives carefully", "elle conduit prudemment"),
    ("she watches birds", "elle regarde les oiseaux"),
    ("she learns pytorch", "elle apprend pytorch"),
    ("she visits museums", "elle visite des musées"),

    ("we study together", "nous étudions ensemble"),
    ("we love programming", "nous aimons la programmation"),
    ("we build models", "nous construisons des modèles"),
    ("we train neural networks", "nous entraînons des réseaux neuronaux"),
    ("we solve problems", "nous résolvons des problèmes"),
    ("we play chess", "nous jouons aux échecs"),
    ("we drink water", "nous buvons de l'eau"),
    ("we read articles", "nous lisons des articles"),
    ("we visit libraries", "nous visitons des bibliothèques"),
    ("we learn data science", "nous apprenons la science des données"),

    ("they study ai", "ils étudient l'ia"),
    ("they build robots", "ils construisent des robots"),
    ("they read papers", "ils lisent des articles"),
    ("they write programs", "ils écrivent des programmes"),
    ("they drink juice", "ils boivent du jus"),
    ("they eat oranges", "ils mangent des oranges"),
    ("they play volleyball", "ils jouent au volley"),
    ("they visit schools", "ils visitent des écoles"),
    ("they watch documentaries", "ils regardent des documentaires"),
    ("they learn transformers", "ils apprennent les transformers"),

    ("my friend likes pizza", "mon ami aime la pizza"),
    ("my sister studies chemistry", "ma sœur étudie la chimie"),
    ("my brother reads books", "mon frère lit des livres"),
    ("my teacher explains lessons", "mon professeur explique les leçons"),
    ("my mother cooks dinner", "ma mère prépare le dîner"),
    ("my father drives carefully", "mon père conduit prudemment"),
    ("our team wins matches", "notre équipe gagne des matchs"),
    ("our project uses pytorch", "notre projet utilise pytorch"),
    ("our class learns python", "notre classe apprend python"),
    ("our university teaches ai", "notre université enseigne l'ia"),

    ("the cat drinks milk", "le chat boit du lait"),
    ("the dog eats meat", "le chien mange de la viande"),
    ("the bird sings loudly", "l'oiseau chante fort"),
    ("the child reads stories", "l'enfant lit des histoires"),
    ("the student solves equations", "l'étudiant résout des équations"),
    ("the professor teaches ai", "le professeur enseigne l'ia"),
    ("the engineer builds robots", "l'ingénieur construit des robots"),
    ("the doctor helps patients", "le médecin aide les patients"),
    ("the artist paints pictures", "l'artiste peint des tableaux"),
    ("the chef cooks delicious food", "le chef prépare de délicieux plats"),

    ("today i study python", "aujourd'hui j'étudie python"),
    ("today we learn ai", "aujourd'hui nous apprenons l'ia"),
    ("every morning i drink coffee", "chaque matin je bois du café"),
    ("every evening she reads books", "chaque soir elle lit des livres"),
    ("every weekend we play football", "chaque week-end nous jouons au football"),
    ("tomorrow they visit paris", "demain ils visitent paris"),
    ("yesterday he watched television", "hier il a regardé la télévision"),
    ("next week we build a model", "la semaine prochaine nous construisons un modèle"),
    ("last year she learned python", "l'année dernière elle a appris python"),
    ("this month they study deep learning", "ce mois-ci ils étudient l'apprentissage profond"),

    ("artificial intelligence changes the world", "l'intelligence artificielle change le monde"),
    ("machine learning improves predictions", "l'apprentissage automatique améliore les prédictions"),
    ("deep learning needs data", "l'apprentissage profond a besoin de données"),
    ("python is easy to learn", "python est facile à apprendre"),
    ("pytorch trains neural networks", "pytorch entraîne des réseaux neuronaux"),
    ("numpy handles arrays", "numpy gère les tableaux"),
    ("students enjoy programming", "les étudiants aiment programmer"),
    ("robots help humans", "les robots aident les humains"),
    ("data science solves problems", "la science des données résout des problèmes"),
    ("technology changes lives", "la technologie change des vies"),

    ("the computer runs quickly", "l'ordinateur fonctionne rapidement"),
    ("the phone rings loudly", "le téléphone sonne fort"),
    ("the train arrives early", "le train arrive tôt"),
    ("the airplane flies high", "l'avion vole haut"),
    ("the sun shines brightly", "le soleil brille vivement"),
    ("the moon looks beautiful", "la lune est belle"),
    ("the stars shine at night", "les étoiles brillent la nuit"),
    ("the flowers smell wonderful", "les fleurs sentent merveilleusement bon"),
    ("the trees grow tall", "les arbres poussent hauts"),
    ("the river flows slowly", "la rivière coule lentement"),

    ("i open the window", "j'ouvre la fenêtre"),
    ("i close the door", "je ferme la porte"),
    ("i clean my room", "je nettoie ma chambre"),
    ("i wash my hands", "je me lave les mains"),
    ("i finish my homework", "je termine mes devoirs"),
    ("i answer the question", "je réponds à la question"),
    ("i solve the puzzle", "je résous le puzzle"),
    ("i enjoy learning", "j'aime apprendre"),
    ("i practice programming", "je pratique la programmation"),
    ("i improve every day", "je m'améliore chaque jour"),

    ("you write beautiful stories", "tu écris de belles histoires"),
    ("you solve difficult problems", "tu résous des problèmes difficiles"),
    ("you study every evening", "tu étudies chaque soir"),
    ("you enjoy reading novels", "tu aimes lire des romans"),
    ("you visit your grandparents", "tu rends visite à tes grands-parents"),
    ("you cook delicious meals", "tu cuisines de délicieux repas"),
    ("you speak french well", "tu parles bien français"),
    ("you understand mathematics", "tu comprends les mathématiques"),
    ("you complete your project", "tu termines ton projet"),
    ("you explore new ideas", "tu explores de nouvelles idées"),

    ("he studies computer science", "il étudie l'informatique"),
    ("he enjoys playing football", "il aime jouer au football"),
    ("he drinks cold water", "il boit de l'eau froide"),
    ("he eats fresh fruit", "il mange des fruits frais"),
    ("he reads scientific papers", "il lit des articles scientifiques"),
    ("he writes clean code", "il écrit du code propre"),
    ("he trains every morning", "il s'entraîne chaque matin"),
    ("he designs websites", "il conçoit des sites web"),
    ("he develops software", "il développe des logiciels"),
    ("he fixes computers", "il répare des ordinateurs"),

    ("she studies data science", "elle étudie la science des données"),
    ("she enjoys classical music", "elle aime la musique classique"),
    ("she cooks healthy food", "elle prépare des repas sains"),
    ("she buys fresh vegetables", "elle achète des légumes frais"),
    ("she drinks orange juice", "elle boit du jus d'orange"),
    ("she teaches young students", "elle enseigne à de jeunes étudiants"),
    ("she develops mobile applications", "elle développe des applications mobiles"),
    ("she designs beautiful paintings", "elle réalise de beaux tableaux"),
    ("she learns neural networks", "elle apprend les réseaux neuronaux"),
    ("she improves her skills", "elle améliore ses compétences"),

    ("we enjoy teamwork", "nous aimons le travail d'équipe"),
    ("we solve complex problems", "nous résolvons des problèmes complexes"),
    ("we build intelligent systems", "nous construisons des systèmes intelligents"),
    ("we create useful software", "nous créons des logiciels utiles"),
    ("we write efficient programs", "nous écrivons des programmes efficaces"),
    ("we test our models", "nous testons nos modèles"),
    ("we improve our accuracy", "nous améliorons notre précision"),
    ("we explore artificial intelligence", "nous explorons l'intelligence artificielle"),
    ("we study machine learning together", "nous étudions l'apprentissage automatique ensemble"),
    ("we enjoy solving challenges", "nous aimons résoudre des défis"),

    ("they travel around the world", "ils voyagent autour du monde"),
    ("they visit famous museums", "ils visitent des musées célèbres"),
    ("they learn new languages", "ils apprennent de nouvelles langues"),
    ("they develop useful applications", "ils développent des applications utiles"),
    ("they solve mathematical problems", "ils résolvent des problèmes mathématiques"),
    ("they train deep learning models", "ils entraînent des modèles d'apprentissage profond"),
    ("they collect useful data", "ils collectent des données utiles"),
    ("they analyze the results", "ils analysent les résultats"),
    ("they improve their projects", "ils améliorent leurs projets"),
    ("they present their work", "ils présentent leur travail"),

    ("my computer is fast", "mon ordinateur est rapide"),
    ("my phone is new", "mon téléphone est neuf"),
    ("my notebook is blue", "mon cahier est bleu"),
    ("my bag is heavy", "mon sac est lourd"),
    ("my bicycle is red", "mon vélo est rouge"),
    ("our laboratory is modern", "notre laboratoire est moderne"),
    ("our classroom is clean", "notre salle de classe est propre"),
    ("our teacher is kind", "notre professeur est gentil"),
    ("our project is successful", "notre projet est réussi"),
    ("our model performs well", "notre modèle fonctionne bien"),

    ("the library is quiet", "la bibliothèque est calme"),
    ("the restaurant serves delicious food", "le restaurant sert de délicieux plats"),
    ("the market opens early", "le marché ouvre tôt"),
    ("the airport is crowded", "l'aéroport est bondé"),
    ("the beach looks beautiful", "la plage est magnifique"),
    ("the hospital is nearby", "l'hôpital est proche"),
    ("the museum is interesting", "le musée est intéressant"),
    ("the university offers many courses", "l'université propose de nombreux cours"),
    ("the classroom has many students", "la salle de classe compte de nombreux étudiants"),
    ("the laboratory contains computers", "le laboratoire contient des ordinateurs"),

    ("python supports machine learning", "python prend en charge l'apprentissage automatique"),
    ("pytorch provides deep learning tools", "pytorch fournit des outils d'apprentissage profond"),
    ("numpy performs fast computations", "numpy effectue des calculs rapides"),
    ("datasets improve model performance", "les ensembles de données améliorent les performances des modèles"),
    ("attention improves translation quality", "l'attention améliore la qualité de la traduction"),
    ("transformers process long sequences", "les transformers traitent de longues séquences"),
    ("embeddings represent words", "les embeddings représentent les mots"),
    ("models learn from data", "les modèles apprennent à partir des données"),
    ("training requires patience", "l'entraînement demande de la patience"),
    ("evaluation measures accuracy", "l'évaluation mesure la précision"),

        ("i visit the beautiful beach", "je visite la belle plage"),
    ("i enjoy sunny weather", "j'aime le temps ensoleillé"),
    ("i buy fresh vegetables", "j'achète des légumes frais"),
    ("i cook delicious pasta", "je prépare de délicieuses pâtes"),
    ("i watch educational videos", "je regarde des vidéos éducatives"),
    ("i complete difficult assignments", "je termine des devoirs difficiles"),
    ("i build intelligent robots", "je construis des robots intelligents"),
    ("i train neural network models", "j'entraîne des modèles de réseaux neuronaux"),
    ("i analyze large datasets", "j'analyse de grands ensembles de données"),
    ("i improve my programming skills", "j'améliore mes compétences en programmation"),

    ("you solve coding challenges", "tu résous des défis de programmation"),
    ("you enjoy learning mathematics", "tu aimes apprendre les mathématiques"),
    ("you study artificial intelligence daily", "tu étudies l'intelligence artificielle chaque jour"),
    ("you create interesting projects", "tu crées des projets intéressants"),
    ("you develop useful software", "tu développes des logiciels utiles"),
    ("you practice deep learning", "tu pratiques l'apprentissage profond"),
    ("you understand neural networks", "tu comprends les réseaux neuronaux"),
    ("you build machine learning models", "tu construis des modèles d'apprentissage automatique"),
    ("you explore data science", "tu explores la science des données"),
    ("you write efficient algorithms", "tu écris des algorithmes efficaces"),

    ("he visits the science museum", "il visite le musée des sciences"),
    ("he enjoys reading history books", "il aime lire des livres d'histoire"),
    ("he studies advanced mathematics", "il étudie les mathématiques avancées"),
    ("he learns artificial intelligence", "il apprend l'intelligence artificielle"),
    ("he develops mobile software", "il développe des logiciels mobiles"),
    ("he trains machine learning models", "il entraîne des modèles d'apprentissage automatique"),
    ("he solves difficult exercises", "il résout des exercices difficiles"),
    ("he explains complex concepts", "il explique des concepts complexes"),
    ("he writes technical reports", "il rédige des rapports techniques"),
    ("he improves model accuracy", "il améliore la précision du modèle"),

    ("she visits the public library", "elle visite la bibliothèque publique"),
    ("she enjoys classical literature", "elle aime la littérature classique"),
    ("she studies computer engineering", "elle étudie le génie informatique"),
    ("she develops artificial intelligence systems", "elle développe des systèmes d'intelligence artificielle"),
    ("she trains deep learning models", "elle entraîne des modèles d'apprentissage profond"),
    ("she analyzes scientific data", "elle analyse des données scientifiques"),
    ("she explains difficult lessons", "elle explique des leçons difficiles"),
    ("she writes research papers", "elle rédige des articles de recherche"),
    ("she improves translation quality", "elle améliore la qualité de la traduction"),
    ("she enjoys solving programming problems", "elle aime résoudre des problèmes de programmation"),

    ("we visit the national museum", "nous visitons le musée national"),
    ("we enjoy learning together", "nous aimons apprendre ensemble"),
    ("we build artificial intelligence applications", "nous construisons des applications d'intelligence artificielle"),
    ("we train transformer models", "nous entraînons des modèles transformers"),
    ("we analyze experimental results", "nous analysons les résultats expérimentaux"),
    ("we develop useful tools", "nous développons des outils utiles"),
    ("we solve real world problems", "nous résolvons des problèmes du monde réel"),
    ("we improve our programming knowledge", "nous améliorons nos connaissances en programmation"),
    ("we practice machine learning every day", "nous pratiquons l'apprentissage automatique chaque jour"),
    ("we enjoy building projects", "nous aimons construire des projets"),

    ("they visit beautiful cities", "ils visitent de belles villes"),
    ("they enjoy learning french", "ils aiment apprendre le français"),
    ("they build intelligent machines", "ils construisent des machines intelligentes"),
    ("they train computer vision models", "ils entraînent des modèles de vision par ordinateur"),
    ("they analyze business data", "ils analysent des données commerciales"),
    ("they solve practical problems", "ils résolvent des problèmes pratiques"),
    ("they develop educational software", "ils développent des logiciels éducatifs"),
    ("they improve system performance", "ils améliorent les performances du système"),
    ("they explore advanced technologies", "ils explorent des technologies avancées"),
    ("they create innovative solutions", "ils créent des solutions innovantes"),

    ("my friend studies robotics", "mon ami étudie la robotique"),
    ("my sister enjoys programming", "ma sœur aime programmer"),
    ("my brother develops games", "mon frère développe des jeux"),
    ("my teacher explains algorithms", "mon professeur explique les algorithmes"),
    ("my father enjoys photography", "mon père aime la photographie"),
    ("my mother grows beautiful flowers", "ma mère cultive de belles fleurs"),
    ("our laboratory develops robots", "notre laboratoire développe des robots"),
    ("our university supports research", "notre université soutient la recherche"),
    ("our project predicts future values", "notre projet prédit les valeurs futures"),
    ("our team wins programming competitions", "notre équipe remporte des compétitions de programmation"),

    ("the engineer designs smart systems", "l'ingénieur conçoit des systèmes intelligents"),
    ("the scientist discovers new ideas", "le scientifique découvre de nouvelles idées"),
    ("the student studies every night", "l'étudiant étudie chaque nuit"),
    ("the professor teaches machine learning", "le professeur enseigne l'apprentissage automatique"),
    ("the doctor examines the patient", "le médecin examine le patient"),
    ("the nurse helps every patient", "l'infirmière aide chaque patient"),
    ("the artist creates beautiful paintings", "l'artiste crée de beaux tableaux"),
    ("the musician plays wonderful songs", "le musicien joue de magnifiques chansons"),
    ("the chef prepares healthy meals", "le chef prépare des repas sains"),
    ("the driver follows traffic rules", "le conducteur respecte le code de la route"),

    ("attention learns word alignment", "l'attention apprend l'alignement des mots"),
    ("transformers outperform recurrent networks", "les transformers surpassent les réseaux récurrents"),
    ("embeddings capture semantic meaning", "les embeddings capturent le sens sémantique"),
    ("language models generate fluent text", "les modèles de langage génèrent un texte fluide"),
    ("datasets contain thousands of examples", "les ensembles de données contiennent des milliers d'exemples"),
    ("training improves neural networks", "l'entraînement améliore les réseaux neuronaux"),
    ("evaluation compares different models", "l'évaluation compare différents modèles"),
    ("beam search improves translation", "la recherche en faisceau améliore la traduction"),
    ("teacher forcing speeds up training", "le teacher forcing accélère l'entraînement"),
    ("artificial intelligence transforms industries", "l'intelligence artificielle transforme les industries"),

        ("i write machine learning code every day", "j'écris du code d'apprentissage automatique chaque jour"),
    ("i practice python programming every evening", "je pratique la programmation python chaque soir"),
    ("i enjoy solving artificial intelligence problems", "j'aime résoudre des problèmes d'intelligence artificielle"),
    ("i build computer vision applications", "je construis des applications de vision par ordinateur"),
    ("i study natural language processing", "j'étudie le traitement du langage naturel"),
    ("i read research papers every week", "je lis des articles de recherche chaque semaine"),
    ("i improve my deep learning models", "j'améliore mes modèles d'apprentissage profond"),
    ("i train transformers on text data", "j'entraîne des transformers sur des données textuelles"),
    ("i evaluate model performance carefully", "j'évalue soigneusement les performances du modèle"),
    ("i enjoy creating innovative solutions", "j'aime créer des solutions innovantes"),

    ("you read programming books every weekend", "tu lis des livres de programmation chaque week-end"),
    ("you enjoy building neural networks", "tu aimes construire des réseaux neuronaux"),
    ("you train computer vision models", "tu entraînes des modèles de vision par ordinateur"),
    ("you analyze customer data", "tu analyses les données des clients"),
    ("you improve translation accuracy", "tu améliores la précision de la traduction"),
    ("you develop artificial intelligence tools", "tu développes des outils d'intelligence artificielle"),
    ("you explore reinforcement learning", "tu explores l'apprentissage par renforcement"),
    ("you understand attention mechanisms", "tu comprends les mécanismes d'attention"),
    ("you implement sequence models", "tu implémentes des modèles de séquence"),
    ("you optimize neural network training", "tu optimises l'entraînement des réseaux neuronaux"),

    ("he develops recommendation systems", "il développe des systèmes de recommandation"),
    ("he studies reinforcement learning", "il étudie l'apprentissage par renforcement"),
    ("he builds language models", "il construit des modèles de langage"),
    ("he analyzes financial data", "il analyse des données financières"),
    ("he improves prediction accuracy", "il améliore la précision des prédictions"),
    ("he writes efficient python programs", "il écrit des programmes python efficaces"),
    ("he trains image classification models", "il entraîne des modèles de classification d'images"),
    ("he evaluates experimental results", "il évalue les résultats expérimentaux"),
    ("he presents technical seminars", "il présente des séminaires techniques"),
    ("he develops useful machine learning libraries", "il développe des bibliothèques utiles d'apprentissage automatique"),

    ("she creates natural language models", "elle crée des modèles de traitement du langage naturel"),
    ("she studies computer vision", "elle étudie la vision par ordinateur"),
    ("she trains speech recognition systems", "elle entraîne des systèmes de reconnaissance vocale"),
    ("she analyzes medical images", "elle analyse des images médicales"),
    ("she improves software quality", "elle améliore la qualité des logiciels"),
    ("she writes clean documentation", "elle rédige une documentation claire"),
    ("she explains neural network architectures", "elle explique les architectures des réseaux neuronaux"),
    ("she develops chatbot applications", "elle développe des applications de chatbot"),
    ("she evaluates translation models", "elle évalue des modèles de traduction"),
    ("she enjoys collaborative research", "elle apprécie la recherche collaborative"),

    ("we build intelligent recommendation systems", "nous construisons des systèmes de recommandation intelligents"),
    ("we study natural language processing together", "nous étudions ensemble le traitement du langage naturel"),
    ("we train transformer based models", "nous entraînons des modèles basés sur les transformers"),
    ("we analyze real world datasets", "nous analysons des ensembles de données du monde réel"),
    ("we improve software performance", "nous améliorons les performances des logiciels"),
    ("we develop machine learning pipelines", "nous développons des pipelines d'apprentissage automatique"),
    ("we optimize training algorithms", "nous optimisons les algorithmes d'entraînement"),
    ("we evaluate multiple deep learning models", "nous évaluons plusieurs modèles d'apprentissage profond"),
    ("we publish our research findings", "nous publions les résultats de nos recherches"),
    ("we enjoy solving practical challenges", "nous aimons résoudre des défis pratiques"),

    ("they create intelligent assistants", "ils créent des assistants intelligents"),
    ("they study advanced deep learning", "ils étudient l'apprentissage profond avancé"),
    ("they build autonomous robots", "ils construisent des robots autonomes"),
    ("they analyze climate data", "ils analysent des données climatiques"),
    ("they improve speech recognition", "ils améliorent la reconnaissance vocale"),
    ("they develop translation software", "ils développent des logiciels de traduction"),
    ("they evaluate artificial intelligence systems", "ils évaluent des systèmes d'intelligence artificielle"),
    ("they optimize transformer architectures", "ils optimisent les architectures transformers"),
    ("they publish scientific articles", "ils publient des articles scientifiques"),
    ("they contribute to open source projects", "ils contribuent à des projets open source"),

    ("my laptop runs machine learning experiments", "mon ordinateur portable exécute des expériences d'apprentissage automatique"),
    ("my project predicts customer behavior", "mon projet prédit le comportement des clients"),
    ("my professor researches artificial intelligence", "mon professeur fait des recherches en intelligence artificielle"),
    ("my classmates enjoy competitive programming", "mes camarades aiment la programmation compétitive"),
    ("my university offers artificial intelligence courses", "mon université propose des cours d'intelligence artificielle"),
    ("our laboratory studies robotics", "notre laboratoire étudie la robotique"),
    ("our software processes large datasets", "notre logiciel traite de grands ensembles de données"),
    ("our model achieves high accuracy", "notre modèle atteint une grande précision"),
    ("our application recognizes speech", "notre application reconnaît la parole"),
    ("our algorithm improves translation quality", "notre algorithme améliore la qualité de la traduction"),

    ("the neural network learns complex patterns", "le réseau neuronal apprend des motifs complexes"),
    ("the transformer processes long sequences efficiently", "le transformer traite efficacement les longues séquences"),
    ("the dataset contains diverse examples", "l'ensemble de données contient des exemples variés"),
    ("the algorithm minimizes prediction error", "l'algorithme minimise l'erreur de prédiction"),
    ("the optimizer updates model parameters", "l'optimiseur met à jour les paramètres du modèle"),
    ("the encoder extracts meaningful features", "l'encodeur extrait des caractéristiques pertinentes"),
    ("the decoder generates fluent translations", "le décodeur génère des traductions fluides"),
    ("the attention mechanism improves alignment", "le mécanisme d'attention améliore l'alignement"),
    ("the vocabulary grows with more data", "le vocabulaire grandit avec davantage de données"),
    ("the model learns from every example", "le modèle apprend de chaque exemple"),

        ("i deploy machine learning models", "je déploie des modèles d'apprentissage automatique"),
    ("i compare different neural networks", "je compare différents réseaux neuronaux"),
    ("i debug python programs", "je débogue des programmes python"),
    ("i explore new datasets", "j'explore de nouveaux ensembles de données"),
    ("i improve translation models", "j'améliore les modèles de traduction"),
    ("i design intelligent systems", "je conçois des systèmes intelligents"),
    ("i implement attention mechanisms", "j'implémente des mécanismes d'attention"),
    ("i evaluate transformer models", "j'évalue des modèles transformers"),
    ("i create useful applications", "je crée des applications utiles"),
    ("i solve real machine learning problems", "je résous de vrais problèmes d'apprentissage automatique"),

    ("you develop computer vision systems", "tu développes des systèmes de vision par ordinateur"),
    ("you train speech recognition models", "tu entraînes des modèles de reconnaissance vocale"),
    ("you evaluate language models", "tu évalues des modèles de langage"),
    ("you improve chatbot responses", "tu améliores les réponses des chatbots"),
    ("you create intelligent assistants", "tu crées des assistants intelligents"),
    ("you understand transformer architecture", "tu comprends l'architecture des transformers"),
    ("you optimize deep learning models", "tu optimises les modèles d'apprentissage profond"),
    ("you analyze image datasets", "tu analyses des ensembles de données d'images"),
    ("you design artificial intelligence projects", "tu conçois des projets d'intelligence artificielle"),
    ("you publish your research", "tu publies tes recherches"),

    ("he develops autonomous vehicles", "il développe des véhicules autonomes"),
    ("he studies natural language processing", "il étudie le traitement du langage naturel"),
    ("he creates intelligent chatbots", "il crée des chatbots intelligents"),
    ("he builds recommendation engines", "il construit des moteurs de recommandation"),
    ("he analyzes healthcare data", "il analyse des données de santé"),
    ("he improves image recognition", "il améliore la reconnaissance d'images"),
    ("he trains large language models", "il entraîne de grands modèles de langage"),
    ("he writes artificial intelligence software", "il écrit des logiciels d'intelligence artificielle"),
    ("he solves complex engineering problems", "il résout des problèmes complexes d'ingénierie"),
    ("he presents research papers", "il présente des articles de recherche"),

    ("she develops intelligent healthcare systems", "elle développe des systèmes de santé intelligents"),
    ("she studies reinforcement learning algorithms", "elle étudie les algorithmes d'apprentissage par renforcement"),
    ("she creates machine learning applications", "elle crée des applications d'apprentissage automatique"),
    ("she analyzes satellite images", "elle analyse des images satellites"),
    ("she improves autonomous robots", "elle améliore les robots autonomes"),
    ("she trains multilingual translation models", "elle entraîne des modèles de traduction multilingues"),
    ("she develops virtual assistants", "elle développe des assistants virtuels"),
    ("she writes research proposals", "elle rédige des propositions de recherche"),
    ("she explains transformer architectures", "elle explique les architectures transformers"),
    ("she enjoys artificial intelligence research", "elle aime la recherche en intelligence artificielle"),

    ("we build multilingual translation systems", "nous construisons des systèmes de traduction multilingues"),
    ("we develop advanced transformer models", "nous développons des modèles transformers avancés"),
    ("we optimize computer vision algorithms", "nous optimisons les algorithmes de vision par ordinateur"),
    ("we analyze healthcare datasets", "nous analysons des ensembles de données médicales"),
    ("we train multilingual language models", "nous entraînons des modèles de langage multilingues"),
    ("we improve speech synthesis systems", "nous améliorons les systèmes de synthèse vocale"),
    ("we publish artificial intelligence research", "nous publions des recherches en intelligence artificielle"),
    ("we collaborate on open source software", "nous collaborons à des logiciels open source"),
    ("we design innovative artificial intelligence solutions", "nous concevons des solutions innovantes d'intelligence artificielle"),
    ("we solve challenging engineering tasks", "nous résolvons des tâches d'ingénierie difficiles"),

    ("they develop autonomous drones", "ils développent des drones autonomes"),
    ("they study multilingual translation", "ils étudient la traduction multilingue"),
    ("they train speech recognition networks", "ils entraînent des réseaux de reconnaissance vocale"),
    ("they evaluate computer vision systems", "ils évaluent des systèmes de vision par ordinateur"),
    ("they optimize artificial intelligence pipelines", "ils optimisent des pipelines d'intelligence artificielle"),
    ("they analyze financial markets", "ils analysent les marchés financiers"),
    ("they improve robotic navigation", "ils améliorent la navigation robotique"),
    ("they build conversational agents", "ils construisent des agents conversationnels"),
    ("they publish innovative research", "ils publient des recherches innovantes"),
    ("they solve difficult real world problems", "ils résolvent des problèmes difficiles du monde réel"),

    ("my research focuses on deep learning", "mes recherches portent sur l'apprentissage profond"),
    ("my university supports artificial intelligence innovation", "mon université soutient l'innovation en intelligence artificielle"),
    ("my laboratory develops multilingual models", "mon laboratoire développe des modèles multilingues"),
    ("our software predicts future outcomes", "notre logiciel prédit les résultats futurs"),
    ("our model recognizes handwritten text", "notre modèle reconnaît le texte manuscrit"),
    ("our project translates multiple languages", "notre projet traduit plusieurs langues"),
    ("our team develops autonomous robots", "notre équipe développe des robots autonomes"),
    ("our algorithm improves speech recognition accuracy", "notre algorithme améliore la précision de la reconnaissance vocale"),
    ("our dataset contains high quality examples", "notre ensemble de données contient des exemples de haute qualité"),
    ("our experiments produce reliable results", "nos expériences produisent des résultats fiables"),

    ("the transformer captures long range dependencies", "le transformer capture les dépendances à longue portée"),
    ("the attention layer focuses on important words", "la couche d'attention se concentre sur les mots importants"),
    ("the encoder learns contextual representations", "l'encodeur apprend des représentations contextuelles"),
    ("the decoder predicts the next token", "le décodeur prédit le jeton suivant"),
    ("the optimizer minimizes training loss", "l'optimiseur minimise la perte d'entraînement"),
    ("the embedding layer represents semantic information", "la couche d'embedding représente les informations sémantiques"),
    ("the neural network generalizes well", "le réseau neuronal généralise bien"),
    ("the translation model generates fluent sentences", "le modèle de traduction génère des phrases fluides"),
    ("the vocabulary contains many unique words", "le vocabulaire contient de nombreux mots uniques"),
    ("artificial intelligence will shape the future", "l'intelligence artificielle façonnera l'avenir")



]

print("Number of sentence pairs:", len(sentence_pairs))

print("\nFirst sentence pair:")
print(sentence_pairs[0])

print("\nEnglish:")
print(sentence_pairs[0][0])

print("\nFrench:")
print(sentence_pairs[0][1])


print("\nAll sentence pairs:\n")

for english, french in sentence_pairs:
    print(f"English: {english}")
    print(f"French : {french}")
    print("-" * 40)

# create vocabularies for English and French
# --------------------------------------------
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


english_vocab = {
    PAD_TOKEN: 0,
    SOS_TOKEN: 1,
    EOS_TOKEN: 2,
    UNK_TOKEN: 3
}

french_vocab = {
    PAD_TOKEN: 0,
    SOS_TOKEN: 1,
    EOS_TOKEN: 2,
    UNK_TOKEN: 3
}

for english, french in sentence_pairs:
    for word in english.split():
        if word not in english_vocab:
            english_vocab[word] = len(english_vocab)

    for word in french.split():
        if word not in french_vocab:
            french_vocab[word] = len(french_vocab)

print("\nEnglish Vocabulary\n")

for word, idx in english_vocab.items():
    print(f"{word:15} -> {idx}")

print("\nFrench Vocabulary\n")

for word, idx in french_vocab.items():
    print(f"{word:15} -> {idx}")


print("\nEnglish Vocabulary Size:", len(english_vocab))
print("French Vocabulary Size :", len(french_vocab))

# converting sentences to numerical representations (tokenization)

def encode_sentence(sentence, vocabulary):

    tokens = sentence.split()

    encoded = [vocabulary[SOS_TOKEN]]

    for token in tokens:

        if token in vocabulary:
            encoded.append(vocabulary[token])
        else:
            encoded.append(vocabulary[UNK_TOKEN])

    encoded.append(vocabulary[EOS_TOKEN])

    return encoded


english = sentence_pairs[0][0]
french = sentence_pairs[0][1]

english_ids = encode_sentence(
    english,
    english_vocab
)

french_ids = encode_sentence(
    french,
    french_vocab
)

print("English:", english)
print("Encoded:", english_ids)

print()

print("French:", french)
print("Encoded:", french_ids)


encoded_english = []

encoded_french = []

for english, french in sentence_pairs:

    encoded_english.append(
        encode_sentence(
            english,
            english_vocab
        )
    )

    encoded_french.append(
        encode_sentence(
            french,
            french_vocab
        )
    )


for i in range(len(encoded_english)):

    print("English IDs:", encoded_english[i])

    print("French IDs :", encoded_french[i])

    print("-"*50)

# building a custom dataset class for translation
class TranslationDataset(Dataset):

    def __init__(
        self,
        source_sentences,
        target_sentences
    ):

        self.source = source_sentences
        self.target = target_sentences

    def __len__(self):

        return len(self.source)

    def __getitem__(self, idx):

        src = torch.tensor(
            self.source[idx],
            dtype=torch.long
        )

        tgt = torch.tensor(
            self.target[idx],
            dtype=torch.long
        )

        return src, tgt

dataset = TranslationDataset(
    encoded_english,
    encoded_french
)
print("Dataset Size:", len(dataset))

src, tgt = dataset[0]

print("Source Tensor:")
print(src)

print()

print("Target Tensor:")
print(tgt)

# collate function for DataLoader to handle variable-length sequences
def collate_fn(batch):

    src_batch = []
    tgt_batch = []

    for src, tgt in batch:

        src_batch.append(src)
        tgt_batch.append(tgt)

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=PAD_IDX
    )

    tgt_batch = pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=PAD_IDX
    )

    return src_batch, tgt_batch

# batch = [

# (torch.tensor([1,4,5,2]),
#  torch.tensor([1,9,10,2])),

# (torch.tensor([1,7,8,9,2]),
#  torch.tensor([1,11,12,13,2])),

# (torch.tensor([1,15,16,17,18,2]),
#  torch.tensor([1,20,21,22,23,2]))
# ]

# src_batch = [x[0] for x in batch]
# tgt_batch = [x[1] for x in batch]

# src_batch = pad_sequence(src_batch,
#                          batch_first=True,
#                          padding_value=PAD_IDX)

# tgt_batch = pad_sequence(tgt_batch,
#                          batch_first=True,
#                          padding_value=PAD_IDX)
# print("Source Batch:")
# print(src_batch)

# print("\nTarget Batch:")
# print(tgt_batch)


# creating a DataLoader for batching and shuffling the data
train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

for src, tgt in train_loader:

    print("Source Batch:")
    print(src)
    print("Shape:", src.shape)
    print()

    print("Target Batch:")
    print(tgt)
    print("Shape:", tgt.shape)
    
print(len(train_loader))  


