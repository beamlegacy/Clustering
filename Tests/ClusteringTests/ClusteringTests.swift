import Nimble
import XCTest
import LASwift
import NaturalLanguage
@testable import Clustering
import Accelerate
import Clustering

// swiftlint:disable:next type_body_length
class ClusteringTests: XCTestCase {
    /// Test adding and removing of data points from a (non-navigation) similarity matrix. For both addition and removal, test that all locations in the matrix (first, last, middle) work as expected
    func testAddandRemoveDataPointsToSimilarityMatrix() throws {
        let cluster = Cluster()
        
        // Checking the state of the matrix after initialization
        expect(cluster.textualSimilarityMatrix.matrix) == Matrix([0])
        
        try cluster.textualSimilarityMatrix.addDataPoint(similarities: [0], type: .page, numExistingNotes: 0, numExistingPages: 0)
        
        // Checking the first page doesn't influence the matrix
        expect(cluster.textualSimilarityMatrix.matrix) == Matrix([0])
        
        try cluster.textualSimilarityMatrix.addDataPoint(similarities: [0.5], type: .page, numExistingNotes: 0, numExistingPages: 1)
        
        // Checking the addition of the second page, first addition to the matrix
        expect(cluster.textualSimilarityMatrix.matrix.flat) == [0, 0.5, 0.5, 0]
        
        try cluster.textualSimilarityMatrix.addDataPoint(similarities: [0.3, 0.2], type: .page, numExistingNotes: 0, numExistingPages: 2)
        
        // Checking the addition of a third page, in position whereToAdd = .last
        expect(cluster.textualSimilarityMatrix.matrix.flat) == [0, 0.5, 0.3, 0.5, 0, 0.2, 0.3, 0.2, 0]
        
        try cluster.textualSimilarityMatrix.addDataPoint(similarities: [0.1, 0.1, 0.1], type: .note, numExistingNotes: 0, numExistingPages: 3)
        
        // Checking the addition of a first note, in position whereToAdd = .first
        expect(cluster.textualSimilarityMatrix.matrix.flat) == [0, 0.1, 0.1, 0.1, 0.1, 0, 0.5, 0.3, 0.1, 0.5, 0, 0.2, 0.1, 0.3, 0.2, 0]
        
        try cluster.textualSimilarityMatrix.addDataPoint(similarities: [0, 0.9, 0.9, 0.9], type: .note, numExistingNotes: 1, numExistingPages: 3)
        
        // Checking the addition of a second note, in position whereToAdd = .middle
        expect(cluster.textualSimilarityMatrix.matrix.flat) == [0, 0, 0.1, 0.1, 0.1, 0, 0, 0.9, 0.9, 0.9, 0.1, 0.9, 0, 0.5, 0.3, 0.1, 0.9, 0.5, 0, 0.2, 0.1, 0.9, 0.3, 0.2, 0]
        
        try cluster.textualSimilarityMatrix.removeDataPoint(index: 3)
        
        // Testing removal from the middle of the matrix
        expect(cluster.textualSimilarityMatrix.matrix.flat) == [0, 0, 0.1, 0.1, 0, 0, 0.9, 0.9, 0.1, 0.9, 0, 0.3, 0.1, 0.9, 0.3, 0]
        
        try cluster.textualSimilarityMatrix.removeDataPoint(index: 0)
        
        // Testing removal of index 0 (first position in the matrix)
        expect(cluster.textualSimilarityMatrix.matrix.flat) == [0, 0.9, 0.9, 0.9, 0, 0.3, 0.9, 0.3, 0]
        
        try cluster.textualSimilarityMatrix.removeDataPoint(index: 2)
        
        // Testing removal of the last position in the matrix
        expect(cluster.textualSimilarityMatrix.matrix.flat) == [0, 0.9, 0.9, 0]
    }

    /// Test the spectral clustering function on a small adjacency matrix
    func testSpectralClustering() throws {
        let cluster = Cluster()
        var i = 0
        var clustersResult = [Int]()
        
        cluster.adjacencyMatrix = Matrix([[0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        repeat {
            let predictedClusters = try cluster.spectralClustering()
            clustersResult = cluster.stabilize(predictedClusters)
            i += 1
        } while clustersResult != [0, 0, 1, 1, 2, 3, 3, 4, 0, 0] && i < 1
        
        // For now it seems that resuls are very stable, if that changes the limit in the loop can be raised up to 10
        expect(clustersResult) == [0, 0, 1, 1, 2, 3, 3, 4, 0, 0]
    }

    /// Test the cosine similarity function
    func testCosineSimilarity() throws {
        let cluster = Cluster()
        let vec1 = [0.0, 1.5, 3.0, 4.5, 6.0]
        let vec2 = [2.0, 4.0, 6.0, 8.0, 10.0]
        let cossim = cluster.cosineSimilarity(vector1: vec1, vector2: vec2)

        expect(cossim).to(beCloseTo(0.9847319278346619, within: 0.0001))
    }
    
    func testGoogleSearchClustering() async throws {
        let cluster = Cluster()
        var UUIDs: [UUID] = []
        
        for _ in 0...8 {
            UUIDs.append(UUID())
        }
        
        let pages = [
            Page(id: UUIDs[0], url: URL(string: "https://www.google.com/search?q=mozart")!, title: "mozart - Google Search"),
            Page(id: UUIDs[1], url: URL(string: "https://www.google.com/search?q=classical%20music%20mozart")!, title: "classical music mozart - Google Search"),
            Page(id: UUIDs[2], url: URL(string: "https://www.google.com/search?q=cat")!, title: "cat - Google Search"),
            Page(id: UUIDs[3], url: URL(string: "https://www.google.com/search?q=dog")!, title: "dog - Google Search"),
            Page(id: UUIDs[4], url: URL(string: "https://www.google.com/search?q=worm")!, title: "worm - Google Search"),
            Page(id: UUIDs[5], url: URL(string: "https://www.google.com/search?q=snake")!, title: "snake - Google Search"),
            Page(id: UUIDs[6], url: URL(string: "https://www.google.com/search?q=beethoven")!, title: "beethoven - Google Search"),
            Page(id: UUIDs[7], url: URL(string: "https://www.google.com/search?q=musique%20classique")!, title: "musique classique - Google Search")
        ]
        var clusteredPageIds: [[UUID]] = []
        
        for page in pages {
            clusteredPageIds = try await cluster.add(page: page).pageGroups
        }
        
        expect(clusteredPageIds.count).to(equal(3))
    }

    /// Test that scoring of textual similarity between two texts is done correctly. At the same opportunity, test all similarity matrices (entities and navigation, in addition to text)
    func testScoreTextualEmbedding() async throws {
        let cluster = Cluster()
        var UUIDs: [UUID] = []
        
        for _ in 0...4 {
            UUIDs.append(UUID())
        }
        
        let pages = [
            Page(id: UUIDs[0], url: URL(string: "https://en.wikipedia.org/wiki/Roger_Federer")!, content: "Federer has played in an era where he dominated men's tennis together with Rafael Nadal and Novak Djokovic, who have been collectively referred to as the Big Three and are widely considered three of the greatest tennis players of all-time. A Wimbledon junior champion in 1998, Federer won his first Grand Slam singles title at Wimbledon in 2003 at age 21. In 2004, he won three out of the four major singles titles and the ATP Finals, a feat he repeated in 2006 and 2007. From 2005 to 2010, Federer made 18 out of 19 major singles finals. During this span, he won his fifth consecutive titles at both Wimbledon and the US Open. He completed the career Grand Slam at the 2009 French Open after three previous runner-ups to Nadal, his main rival up until 2010. At age 27, he also surpassed Pete Sampras's then-record of 14 Grand Slam men's singles titles at Wimbledon in 2009."),
            Page(id: UUIDs[1], url: URL(string: "https://en.wikipedia.org/wiki/Rafael_Nadal")!, content: "From childhood through most of his professional career, Nadal was coached by his uncle Toni. He was one of the most successful teenagers in ATP Tour history, reaching No. 2 in the world and winning 16 titles before his 20th birthday, including his first French Open and six Masters events. Nadal became No. 1 for the first time in 2008 after his first major victory off clay against his rival, the longtime top-ranked Federer, in a historic Wimbledon final. He also won an Olympic gold medal in singles that year in Beijing. After defeating Djokovic in the 2010 US Open final, the 24-year-old Nadal became the youngest man in the Open Era to achieve the career Grand Slam, and also became the first man to win three majors on three different surfaces (hard, grass and clay) the same calendar year. With his Olympic gold medal, he is also one of only two male players to complete the career Golden Slam."),
            Page(id: UUIDs[2], url: URL(string: "https://fr.wikipedia.org/wiki/Roger_Federer")!, content: "Sa victoire à Roland-Garros en 2009 lui a permis d'accomplir le Grand Chelem en carrière sur quatre surfaces différentes. En s'adjugeant ensuite l'Open d'Australie en 2010, il devient le premier joueur de l'histoire à avoir conquis l'ensemble de ses titres du Grand Chelem sur un total de cinq surfaces, depuis le remplacement du Rebound Ace australien par une nouvelle surface : le Plexicushion. Federer a réalisé le Petit Chelem de tennis à trois reprises, en 2004, 2006 et 2007, ce qui constitue à égalité avec Novak Djokovic, le record masculin toutes périodes confondues. Il est ainsi l'unique athlète à avoir gagné trois des quatre tournois du Grand Chelem deux années successives. Il atteint à trois reprises, et dans la même saison, les finales des quatre tournois majeurs, en 2006, 2007 et 2009, un fait unique dans l'histoire de ce sport."),
            Page(id: UUIDs[3], url: URL(string: "https://fr.wikipedia.org/wiki/Rafael_Nadal")!, content: "Il est considéré par tous les spécialistes comme le meilleur joueur sur terre battue de l'histoire du tennis, établissant en effet des records majeurs, et par la plupart d'entre eux comme l'un des meilleurs joueurs de simple de tous les temps, si ce n’est le meilleur. Il a remporté vingt tournois du Grand Chelem (un record qu'il détient avec Roger Federer et Novak Djokovic) et est le seul joueur à avoir remporté treize titres en simple dans un de ces quatre tournois majeurs : à Roland-Garros où il s'est imposé de 2005 à 2008, de 2010 à 2014, puis de 2017 à 2020. À l'issue de l'édition 2021, où il est détrôné en demi-finale par Novak Djokovic, il présente un bilan record de cent-cinq victoires pour trois défaites dans ce tournoi parisien, et ne compte aucune défaite en finale. Il a remporté également le tournoi de Wimbledon en 2008 et 2010, l'Open d'Australie 2009 et l'US Open 2010, 2013, 2017 et 2019. Il est ainsi le septième joueur de l'histoire du tennis à réaliser le « Grand Chelem en carrière » en simple. À ce titre, Rafael Nadal est le troisième joueur et le plus jeune à s'être imposé durant l'ère Open dans les quatre tournois majeurs sur quatre surfaces différentes, performance que seuls Roger Federer, Andre Agassi et Novak Djokovic ont accomplie."),
            Page(id: UUIDs[4], url: URL(string: "https://www.youtube.com")!, content: "All")
        ]
        var clusteredPageIds: [[UUID]] = []
        
        for page in pages {
            clusteredPageIds = try await cluster.add(page: page).pageGroups
        }
        
        expect(clusteredPageIds.count).to(equal(2))
    }
    
    func testLongAndShortText() async throws {
        let cluster = Cluster()
        var UUIDs: [UUID] = []
        
        for _ in 0...5 {
            UUIDs.append(UUID())
        }
        
        let pages = [
            Page(id: UUIDs[0], url: URL(string: "https://www.amazon.fr/Nvidia-SHIELD-Support-Vendu-S%C3%A9par%C3%A9ment/dp/B07Z6RD4M9/ref=sr_1_1?keywords=shield+tv+pro+2022&qid=1649420635&sprefix=%2Caps%2C47&sr=8-1")!, title: "Nvidia Shield Tv Pro (Support Shield Vendu Separement) : Amazon.Fr: High-Tech", content: "Je possédais la version précédente (2017) et c’est naturellement que je me suis orienté vers la 2019 qui est plus rapide grâce à son processeur Nvidia Tegra X1+. L’expérience 4K HDR est au top que ce soit pour regarder des films, des séries ou encore jouer à des jeux. Cette version pro est compatible Dolby Vision HDR, Dolby Audio et HDR10 ce qui est un réel plus. Il est important de préciser que la version Pro dispose de 1 Go de RAM supplémentaire que la version classique et elle dispose aussi de 16 Go de stockage interne pour 8 Go pour la version classique.Points forts :- Design- Fluidité- Facile à configurer- Simplicité de l’interface- Capable de lire n'importe quel fichier vidéo ou de musique- Lecture multimédia au top- Le Bluetooth est utilisable pour connecter n'importe quel périphérique bluetooth- Reconnaissance vocale de qualité- Gestion HDR- 3 Go de mémoire vive et de 16 Go d'espace de stockage- 2 ports USB 3.0- Télécommande aux touches rétroéclairées, télécommande que l’on peut transformer en télécommande universelle pour contrôler le téléviseur. Accès direct à Netflix. Elle est compatible avec les versions précédentes. Il y a aussi une fonction de localisation à distance ce qui est bien pratique.- Bon Système d'upscaling- Google Assistant et Chromecast 4K intégré qui permet d’envoyer très facilement du contenu du téléphone mobile vers la télévision.Points faibles :- Pas de port MicroSD- Absence d’alimentation USB- Prix- Alimentation intégrée- Les 2 ports USB 3.0 sont trop proches l’un de l’autre, pas toujours pratique en fonction de ce que l’on branche. Il faudra donc brancher un Hub.- Play Store réduit.Ne mérite pas 5 étoiles car vu le prix, Nvidia aurait quand même pu proposer plus d'évolutions à son boitier : alimentation USB, wifi 6, plus d'espace de stockage que les 16Go, port MicroSD... J'ai achete sur la base d'articles annoncant la nouvelle version\"\"plus puissante, etc..\"\" Je cherchais une alternative a mon player Popcorn, qui puisse supporte le 4k,HDR, x265 et un meilleur audio. Mon utilisation principale est d'exploiter une base considerable de films et de CD. Je ne dirai rien de la plateforme de jeux, ca n'est pas mon truc, (sauf que je confirme que l'acceleration de perf avec le changement de modele 2019(et les 100 EUR) nes'applique qu'au Gaming et pas du tout a l'environnement android) Mes films sont organises rationellement, sur mon NAS, geres par un Librarian ecrit par moi meme sur mon PC, et executable par le PopCorn. La musique est geree par lePopCorn, plutot bien, mais le rendu audible est perfectible. Donc j'ai opte pour ce monde Android que je croyais connaitre et pour lequel j'ai programme en un temps.. mais ce que je n'avais pas anticipe c'est que Android TV , du moins tel qu'il est implemente sur le Shield, est une camisole de force contraignante, penible et pour tout dire CH..NTE au max! Apres avoir gache deux Week ends sur Internet et You Tube pour tenter de deserrer les contraintes j'hesite encore a retourner ma Shield.. Ca commence par un constat objectif: J'ai une machine pseudo-android qui utilise un espace 3840x2160 pixels au travers d'une interface simpliste digne d'un commodore avec ses gros boutons non redimensionnables, une souris absente, donc un appareil infirme au niveau de l'interface utilisateur! DRAMATIQUE! Ca continue quand on decouvre que l'on a pas droit non plus a un Browser internet decent! Pas de Chrome sur google et les substituts sont PATHETIQUES, mon troisieme Cellphone a 80 Euros fait nettement mieux! Pas question d'installer les applis de l' Android Store, on accede a des sous-store orientees perception d'abonnement et de droits, C'est la frustation TOTALE. Le support et la doc online sont inacessibles depuis la machine, donc on lit sur son tel avec chrome ou opera ou dolphin ou ou ou..la miserable douzaine de pages fournies par Nvidia.. Pas de file Manager non plus sauf a chercher dans les coins.. Pas non plus de gestionnaire d'acces NAS au niveau du Shield, cela est remis a chaque application avec des inconsequences selon que l'on parle via Kodi, ou Plex, ou VLC Et puis on se decide a tenter de Hacker pour faire sauter ces limitations frustrantes, castratrices et de fait INADMISSIBLES. Alors on se lance dans la centaine de video sur You Tube decrivant des work-around pour ameliorer la situation. La plupart sont inutiles, s'appliquent a la version precedente de Shield. Et il me semble bien qu'une des raisons de cette nouvelle version est de limiter le hacking car bientot les applis nouvellement installees via une appstore nonofficielle sont flagees par le Shield.. Bref je suis nul, sansdoute, et pas content, surement! . Car je me sens encore PLUS FRUSTRE que quand APPLE m'enferme dans ses systemes, car c'est MIEUX FAIT chez la Pomme! Pour que j'en sois a dire cela c'est vraiment que la situation est penible. Doncsi c'est pour jouer, je suis incompetent. si c'est pour regarder netflix ou myCanal (!) ca colle sans doute Si c'est pour installer un vrai Media center ca n'ira que si vous etes gentil, discipline pas exigeant et la tete bien courbee pour obeir a google TV, interface et OS de m..."),
            Page(id: UUIDs[1], url: URL(string: "https://www.frandroid.com/bons-plans/1216689_la-nvidia-shield-tv-pro-est-de-retour-en-promotion-juste-avant-la-fin-des-soldes")!, title: "L'excellente Nvidia Shield Tv Pro Baisse Son Prix Pour La Fin Des Soldes", content: "La Nvidia Shield TV Pro continue de trôner au sommet des box Android TV pour plusieurs raisons. Tout d’abord, c’est un boîtier multimédia complet au niveau du divertissement, que ce soit le streaming avec les applications phares telles que Netflix et autres ou encore le jeu vidéo, notamment via le service GeForce Now accessible en exclusivité sur la machine. Ensuite, c’est une box qui est encore mis à jour à l’heure actuelle, puisqu’elle tourne maintenant sous Android 11. Bref, c’est un must-have, surtout lorsqu’une promotion pointe le bout de son nez. Pour jouer, la Nvidia Shield TV Pro (2019) est une plateforme idéale afin de profiter de l’ensemble des jeux disponibles sur le Play Store. Sa puce Tegra X1+ épaulée par 3 Go de mémoire vive est 25 % plus puissante par rapport à la configuration de l’ancien modèle, pour une expérience gaming encore plus agréable. Pour aller plus loin avec des graphismes toujours plus beaux, notez qu’il est également possible d’accéder (gratuitement ou moyennant un abonnement) au service de cloud gaming GeForce Now, d’autant plus que la box est parfaitement optimisée pour. Pour améliorer votre débit, vous pourrez évidemment brancher un câble Ethernet et il y a même plusieurs ports USB pour connecter des manettes si toutes ne sont pas compatibles Bluetooth. En ce qui concerne les films et séries, la box de Nvidia propose les certifications Dolby Vision et Dolby Atmos, avec également une compatibilité 4K grâce à un nouvel algorithme plus poussé qu’auparavant, dans le but d’obtenir une meilleure qualité d’images lorsque vous visionnerez du contenu. Vous pourrez alors en profiter sur les différentes applications de streaming, mais aussi avec vos fichiers compatibles via le port USB de la Shield TV Pro, via un disque dur externe ou autre. On peut donc s’en servir comme serveur Plex en Full HD. Les données transmises par le biais de ce formulaire sont destinées à HUMANOID, société éditrice du site Frandroid en sa qualité de responsable de traitement. Elles ne seront en aucun cas cédées à des tiers. Ces données sont traitées sous réserve d'obtention de votre consentement pour vous envoyer par e-mail des actualités et informations relatives aux contenus éditoriaux publiés sur Frandroid. Vous pouvez vous opposer à tout moment à ces e-mails en cliquant sur les liens de désinscriptions présents dans chacun d'eux. Pour plus d'informations, vous pouvez consulter l'intégralité de notre politique de traitement de vos données personnelles. Vous disposez d'un droit d'accès, de rectification, d'effacement, de limitation, de portabilité et d'opposition pour motif légitime aux données personnelles vous concernant. Pour exercer l'un de ces droits, merci d'effectuer votre demande via notre formulaire de demandes d'exercices de droits dédié."),
            Page(id: UUIDs[2], url: URL(string: "https://www.nvidia.com/en-us/shield/")!, title: "Shield Tv 4K Hdr", content: "SHIELD TV is compact, stealth, and designed to disappear behind your entertainment center, right along with your cables. SHIELD Pro takes this performance to the next level. It’s built for the most demanding users and beautifully designed to be the perfect centerpiece of your entertainment center. Search for movies and shows, access media playback controls, change the temperature, dim the lights, and so much more. Ask questions and see the answers on your TV, including Google Photos, your calendar, sports scores, and more. Smart devices such as thermostats, doorbells, cameras, even coffee makers, can be controlled with your voice and set on schedules to make your life easier†‡. Even automate and control your smart devices‡. The Google Assistant is always on, ready to help, and makes life in your living room that much more relaxing*†."),
            Page(id: UUIDs[3], url: URL(string: "https://www.google.com/search?q=bloodborne%20sculpts")!, title: "Bloodborne Sculpts"),
            Page(id: UUIDs[4], url: URL(string: "https://www.etsy.com/market/bloodborne_sculpture?ref=seller_tag_bottom_text-1")!, title:"Bloodborne Sculpture | Etsy France"),
            Page(id: UUIDs[5], url: URL(string: "https://en.wikipedia.org/wiki/Bloodborne")!, title:"Bloodborne", content: "Bloodborne[b] is a 2015 action role-playing game developed by FromSoftware and published by Sony Computer Entertainment for the PlayStation 4. Bloodborne follows the player's character, a Hunter, through the decrepit Gothic, Victorian-era–inspired city of Yharnam, whose inhabitants are afflicted with a blood-borne disease. Attempting to find the source of the plague, the player's character unravels the city's mysteries while fighting beasts and cosmic beings. The game is played from a third-person perspective. Players control a customizable protagonist, and the gameplay is focused on strategic weapons-based combat and exploration. Players battle varied enemies while using items such as swords and firearms, exploring different locations, interacting with non-player characters, and unravelling the city's mysteries. Bloodborne began development in 2012 under the working title of Project Beast. Bearing many similarities to the Souls series of games by the same developer and director, Bloodborne was inspired by the literary works of authors H. P. Lovecraft and Bram Stoker and the architectural design of real-world locations in countries such as Romania, the Czech Republic and Scotland. Bloodborne was released in March 2015 and received critical acclaim, with praise directed at its gameplay, particularly its high difficulty level, atmosphere, sound design, Lovecraftian themes, and interconnected world design. Some criticism was directed at its technical performance at launch, which was improved with post-release updates. An expansion adding additional content, The Old Hunters, was released in November 2015. By the end of 2015, the game had sold over two million copies worldwide. Bloodborne won several awards and has been cited as one of the greatest video games ever made. Some related media and adaptations have also been released, including a card game, board game and comic book series. Bloodborne is an action role-playing game played from a third-person perspective and features elements similar to those found in the Souls series of games, particularly Demon's Souls and Dark Souls.[1] The player makes their way through different locations within the decrepit Gothic world of Yharnam while battling varied enemies, including bosses,[2] collecting different types of items that have many uses, interacting with non-player characters,[3] opening up shortcuts, and continuing through the main story. At the beginning of the game, the player creates their character, the Hunter. The player determines the basic details of the Hunter; gender, hairstyle, name, skin colour, body shape, voice, and eye colour are some of the options the player can customize. The player also chooses a starting class, known as an \"\"Origin\"\", which provides a basic backstory for the Hunter and sets the player's starting attributes. The origins, while describing the player character's past, do not have any effect on gameplay beyond altering starting stats.[4][5] Another way the player defines their Hunter is by choosing what brotherhood they are a member of. These religious societies, known as \"\"Covenants\"\", each have their views on the world of Yharnam.[6][7] The player can return to the safe zone, known as the \"\"Hunter's Dream\"\", by interacting with lanterns spread throughout the world of Yharnam. Doing so replenishes health, but repopulates all enemies in the game world, with the exception of bosses and mini-bosses. Lanterns also serve as the game's checkpoints; the player will return to the last activated lantern when they die. Positioned separate from Yharnam, the Hunter's Dream delivers some of the game's basic features to the player. Players may purchase helpful items, such as weapons, clothing and consumables, from the Messengers using Blood Echoes or Insight, level up their character by talking to the Doll, or upgrade their weapons in the workshop, among other things. Unlike Yharnam and all other locations in the game, the Hunter's Dream is considered completely safe as it is the only location in the game not to feature enemies. However, the last two boss battles of the game take place in the Hunter's Dream, although both are optional to the player.[8][9][10]")
        ]
        
        var clusteredPageIds: [[UUID]] = []
        
        for page in pages {
            clusteredPageIds = try await cluster.add(page: page).pageGroups
        }
        
        expect(clusteredPageIds.count).to(equal(3))
    }
    
    func testMixPageNote() async throws {
        let cluster = Cluster()
        var pageUUIDs: [UUID] = []
        var noteUUIDs: [UUID] = []
        
        for _ in 0...2 {
            pageUUIDs.append(UUID())
            noteUUIDs.append(UUID())
        }
        
        let pages = [
            Page(id: pageUUIDs[0], url: URL(string: "https://en.wikipedia.org/wiki/Roger_Federer")!, content: "Federer has played in an era where he dominated men's tennis together with Rafael Nadal and Novak Djokovic, who have been collectively referred to as the Big Three and are widely considered three of the greatest tennis players of all-time. A Wimbledon junior champion in 1998, Federer won his first Grand Slam singles title at Wimbledon in 2003 at age 21. In 2004, he won three out of the four major singles titles and the ATP Finals, a feat he repeated in 2006 and 2007. From 2005 to 2010, Federer made 18 out of 19 major singles finals. During this span, he won his fifth consecutive titles at both Wimbledon and the US Open. He completed the career Grand Slam at the 2009 French Open after three previous runner-ups to Nadal, his main rival up until 2010. At age 27, he also surpassed Pete Sampras's then-record of 14 Grand Slam men's singles titles at Wimbledon in 2009."),
            Page(id: pageUUIDs[1], url: URL(string: "https://en.wikipedia.org/wiki/Rafael_Nadal")!, content: "From childhood through most of his professional career, Nadal was coached by his uncle Toni. He was one of the most successful teenagers in ATP Tour history, reaching No. 2 in the world and winning 16 titles before his 20th birthday, including his first French Open and six Masters events. Nadal became No. 1 for the first time in 2008 after his first major victory off clay against his rival, the longtime top-ranked Federer, in a historic Wimbledon final. He also won an Olympic gold medal in singles that year in Beijing. After defeating Djokovic in the 2010 US Open final, the 24-year-old Nadal became the youngest man in the Open Era to achieve the career Grand Slam, and also became the first man to win three majors on three different surfaces (hard, grass and clay) the same calendar year. With his Olympic gold medal, he is also one of only two male players to complete the career Golden Slam."),
            Page(id: pageUUIDs[2], url: URL(string: "https://www.youtube.com")!, content: "All")
        ]
        
        let notes = [
            ClusteringNote(id: noteUUIDs[0], title: "Roger Federer", content: "Sa victoire à Roland-Garros en 2009 lui a permis d'accomplir le Grand Chelem en carrière sur quatre surfaces différentes. En s'adjugeant ensuite l'Open d'Australie en 2010, il devient le premier joueur de l'histoire à avoir conquis l'ensemble de ses titres du Grand Chelem sur un total de cinq surfaces, depuis le remplacement du Rebound Ace australien par une nouvelle surface : le Plexicushion. Federer a réalisé le Petit Chelem de tennis à trois reprises, en 2004, 2006 et 2007, ce qui constitue à égalité avec Novak Djokovic, le record masculin toutes périodes confondues. Il est ainsi l'unique athlète à avoir gagné trois des quatre tournois du Grand Chelem deux années successives. Il atteint à trois reprises, et dans la même saison, les finales des quatre tournois majeurs, en 2006, 2007 et 2009, un fait unique dans l'histoire de ce sport."),
            ClusteringNote(id: noteUUIDs[1], title: "Rafael Nadal", content: "Il est considéré par tous les spécialistes comme le meilleur joueur sur terre battue de l'histoire du tennis, établissant en effet des records majeurs, et par la plupart d'entre eux comme l'un des meilleurs joueurs de simple de tous les temps, si ce n’est le meilleur. Il a remporté vingt tournois du Grand Chelem (un record qu'il détient avec Roger Federer et Novak Djokovic) et est le seul joueur à avoir remporté treize titres en simple dans un de ces quatre tournois majeurs : à Roland-Garros où il s'est imposé de 2005 à 2008, de 2010 à 2014, puis de 2017 à 2020. À l'issue de l'édition 2021, où il est détrôné en demi-finale par Novak Djokovic, il présente un bilan record de cent-cinq victoires pour trois défaites dans ce tournoi parisien, et ne compte aucune défaite en finale. Il a remporté également le tournoi de Wimbledon en 2008 et 2010, l'Open d'Australie 2009 et l'US Open 2010, 2013, 2017 et 2019. Il est ainsi le septième joueur de l'histoire du tennis à réaliser le « Grand Chelem en carrière » en simple. À ce titre, Rafael Nadal est le troisième joueur et le plus jeune à s'être imposé durant l'ère Open dans les quatre tournois majeurs sur quatre surfaces différentes, performance que seuls Roger Federer, Andre Agassi et Novak Djokovic ont accomplie.")
        ]
        
        var clusteredPageIds: [[UUID]] = []
        var clusteredNoteIds: [[UUID]] = []
        
        for page in pages {
            (clusteredPageIds, clusteredNoteIds, _) = try await cluster.add(page: page)
        }
        
        for note in notes {
            (clusteredPageIds, clusteredNoteIds, _) = try await cluster.add(note: note)
        }
        
        expect(clusteredPageIds.count).to(equal(4))
        expect(clusteredNoteIds.count).to(equal(4))
    }

    /// Test that similarities between notes/active-sources to suggestions are returned correctly, for the sake of correct scoring of suggestions
    func testCreateSimilarities() throws {
        let cluster = Cluster()
        
        cluster.textualSimilarityMatrix.matrix = Matrix([[0, 0, 0, 0.9, 0.8, 0.7],
                                                         [0, 0, 0, 0.5, 0.5, 0.5],
                                                         [0, 0, 0, 0.1, 0.2, 0.3],
                                                         [0.9, 0.5, 0.1, 0, 0.5, 0.2],
                                                         [0.8, 0.5, 0.2, 0.5, 0, 0.3],
                                                         [0.7, 0.5, 0.3, 0.2, 0.3, 0]])
        cluster.notes = [ClusteringNote(id: UUID(), title: "First note", content: "note"),
                         ClusteringNote(id: UUID(), title: "Second note", content: "note"),
                         ClusteringNote(id: UUID(), title: "Third note", content: "note")
        ]
        cluster.pages = [Page(id: UUID(), url: URL(string: "http://firtpage.com")!, title: "First page", content: "page"),
                         Page(id: UUID(), url: URL(string: "http://secondpage.com")!, title: "Second page", content: "page"),
                         Page(id: UUID(), url: URL(string: "http://thirdpage.com")!, title: "Third page", content: "page")
        ]
        
        let noteGroups = [[cluster.notes[0].id], [cluster.notes[1].id], [cluster.notes[2].id], [], []]
        let pageGroups = [[], [cluster.pages[2].id], [], [cluster.pages[0].id, cluster.pages[1].id]]
        let mySimilarities = cluster.createSimilarities(pageGroups: pageGroups, noteGroups: noteGroups)
        
        expect(mySimilarities[cluster.notes[0].id]) == [:]
        expect(mySimilarities[cluster.notes[1].id]) == [cluster.pages[2].id: 0.5]
        expect(mySimilarities[cluster.notes[2].id]) == [:]
        expect(mySimilarities[cluster.pages[0].id]) == [cluster.pages[1].id: 0.5]
    }
    
    /// Test for the getSubmatrix method
    func testGetSubMatrix() throws {
        let cluster = Cluster()
        let myMatrix = Matrix([[0, 1, 2, 3],
                              [4, 5, 6, 7],
                              [8, 9, 10, 11],
                              [12, 13, 14, 15]])
        
        expect(try cluster.getSubmatrix(of: myMatrix, withIndeces:[0, 2]).flat) == [0.0, 2.0, 8.0, 10.0]
    }
    
    /// Test that two notes alone in a subgroup are separated correctly
    func testSeparateTwoNotesOnly() throws {
        let cluster = Cluster()
        let adjacencyOnlyNotes = Matrix([[0, -1],
                                        [-1, 0]])
        let result = try cluster.spectralClustering(on: adjacencyOnlyNotes, numGroups: 2, numNotes: 2)
        
        expect(Set(result)) == Set([0, 1])
    }
    
    /// Test that two notes in a group with pages are separated correctly
    func testSeparateNotesWithPages() throws {
        let cluster = Cluster()
        let adjacencySubgroup = Matrix([[0, -1, 0.5, 0.5],
                                        [-1, 0, 1, 0],
                                        [0.5, 1, 0, 0.2],
                                        [0.5, 0, 0.2, 0]])
        let result = try cluster.spectralClustering(on: adjacencySubgroup, numGroups: 2, numNotes: 2)
        
        expect(Set(result)) == Set([0, 1])
        expect(result[0]) != result[1]
    }
    
    func testRemoveNote() async throws {
        let cluster = Cluster()
        var noteUUIDs: [UUID] = []
        var clusteredNoteIds: [[UUID]] = []
        
        for i in 0...2 {
            noteUUIDs.append(UUID())
            let myNote = ClusteringNote(id: noteUUIDs[i], title: "My note", content: "This is my note")
            (_, clusteredNoteIds, _) = try await cluster.add(note: myNote)
            
        }
        
        expect(clusteredNoteIds[0].count) == 3
        expect(clusteredNoteIds[0][0]) == noteUUIDs[0]
        expect(clusteredNoteIds[0][1]) == noteUUIDs[1]
        expect(clusteredNoteIds[0][2]) == noteUUIDs[2]
        
        (_, clusteredNoteIds, _) = try await cluster.removeNote(noteId: noteUUIDs[0])
        
        expect(clusteredNoteIds[0].count) == 2
        expect(clusteredNoteIds[0][0]) == noteUUIDs[1]
        expect(clusteredNoteIds[0][1]) == noteUUIDs[2]
    }
    
    func testRemovePage() async throws {
        let cluster = Cluster()
        // Here we don't want to test that notes with little content are not added
        var pageUUIDs: [UUID] = []
        var clusteredPageIds: [[UUID]] = []
        
        for i in 0...2 {
            pageUUIDs.append(UUID())
            let myPage = Page(id: pageUUIDs[i], url: URL(string: "http://note.com")!, title: "My note", content: "This is my note")
            (clusteredPageIds, _, _) = try await cluster.add(page: myPage)
            
        }
        
        expect(clusteredPageIds[0].count) == 3
        expect(clusteredPageIds[0][0]) == pageUUIDs[0]
        expect(clusteredPageIds[0][1]) == pageUUIDs[1]
        expect(clusteredPageIds[0][2]) == pageUUIDs[2]
        
        (clusteredPageIds, _, _) = try await cluster.removePage(pageId: pageUUIDs[0])
        
        expect(clusteredPageIds[0].count) == 2
        expect(clusteredPageIds[0][0]) == pageUUIDs[1]
        expect(clusteredPageIds[0][1]) == pageUUIDs[2]
    }
    
    func testGetExportInformationForIdPage() async throws {
        let cluster = Cluster()
        let pageId = UUID()
        let myPage = Page(id: pageId, url: URL(string: "http://roger.com")!, title: "Roger Federer", content: "He was born on 8 August 1981 in Basel.")
        
        _ = try await cluster.add(page: myPage)
        
        let pageInformation = cluster.getExportInformationForId(id: pageId)
        let expectedInformation = InformationForId(title: "Roger Federer", content: "He was born on 8 August 1981 in Basel.")
        
        expect(pageInformation) == expectedInformation
        
        let emptyInformation = cluster.getExportInformationForId(id: UUID())
        
        expect(emptyInformation) == InformationForId()
    }

    func testGetInformationForIdNote() async throws {
        let cluster = Cluster()
        let noteId = UUID()
        let myNote = ClusteringNote(id: noteId, title: "Roger Federer", content: "Federer has played in an era where he dominated men's tennis along with Rafael Nadal and Novak Djokovic. Referred to as the Big Three, they are considered by some to be the three greatest tennis players of all time.[c] A Wimbledon junior champion in 1998, Federer won his first major singles title at Wimbledon in 2003 at age 21. In 2004, he won three of the four major singles titles and the ATP Finals,[d] a feat he repeated in 2006 and 2007. From 2005 to 2010, he made 18 out of 19 major singles finals. During this span, he won five consecutive titles at both Wimbledon and the US Open. He completed the career Grand Slam at the 2009 French Open after three previous runner-up finishes to Nadal, his main rival until 2010. At age 27, he surpassed Pete Sampras's record of 14 major men's singles titles at Wimbledon in 2009.")
        
        _ = try await cluster.add(note: myNote)
        
        let noteInformation = cluster.getExportInformationForId(id: noteId)
        let expectedInformation = InformationForId(title: "Roger Federer", content: "Federer has played in an era where he dominated men\'s tennis along with Rafael Nadal and Novak Djokovic. Referred to as the Big Three, they are considered by some to be the three greatest tennis players of all time.[c] A Wimbledon junior champion in 1998, Federer won his first major singles title at Wimbledon in 2003 at age 21. In 2004, he won three of the four major singles titles and the ATP Finals,[d] a feat he repeated in 2006 and 2007. From 2005 to 2010, he made 18 out of 19 major singles finals. During this span, he won five consecutive titles at both Wimbledon and the US Open. He completed the career Grand Slam at the 2009 French Open after three previous runner-up finishes to Nadal, his main rival until 2010. At age 27, he surpassed Pete Sampras\'s record of 14 major men\'s singles titles at Wimbledon in 2009.")
        
        expect(noteInformation) == expectedInformation
        
        let emptyInformation = cluster.getExportInformationForId(id: UUID())
        
        expect(emptyInformation) == InformationForId()
    }
    
    // swiftlint:disable:next file_length
}
