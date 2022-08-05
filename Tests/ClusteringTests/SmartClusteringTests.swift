import Nimble
import XCTest
@testable import Clustering


class SmartClusteringTests: XCTestCase {
    func testConcurrentAdd() async throws {
        let cluster = SmartClustering()
        let exp = expectation(description: "Add")
        
        cluster.prepare()
        
        let textualItems = [
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=mozart", title: "mozart - Google Search", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=classical%20music%20mozart", title: "classical music mozart - Google Search", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=cat", title: "cat - Google Search", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=dog", title: "dog - Google Search", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=worm", title: "worm - Google Search", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=snake", title: "snake - Google Search", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=beethoven", title: "beethoven - Google Search", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=musique%20classique", title: "musique classique - Google Search", type: TextualItemType.page)
        ]
        
        for texualItem in textualItems {
            Task {
                _ = try await cluster.add(textualItem: texualItem)
            }
        }
        
        sleep(1)
        exp.fulfill()
        
        await waitForExpectations(timeout: 2)
        
        expect(cluster.textualItems.count).to(equal(8))
    }
    
    func testAddBeforePrepareEnds() async throws {
        let cluster = SmartClustering()
        let exp = expectation(description: "Add")
        let textualItems = [
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=mozart", title: "mozart - Google Search", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=classical%20music%20mozart", title: "classical music mozart - Google Search", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=cat", title: "cat - Google Search", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=dog", title: "dog - Google Search", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=worm", title: "worm - Google Search", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=snake", title: "snake - Google Search", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=beethoven", title: "beethoven - Google Search", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=musique%20classique", title: "musique classique - Google Search", type: TextualItemType.page)
        ]
        
        Task {
            cluster.prepare()
        }
        
        for texualItem in textualItems {
            Task {
                _ = try await cluster.add(textualItem: texualItem).pageGroups
            }
        }
        
        sleep(1)
        exp.fulfill()
        
        await waitForExpectations(timeout: 2)
        
        expect(cluster.textualItems.count).to(equal(8))
    }
    
    func testGoogleSearchClusteringWithChangingThreshold() async throws {
        let cluster = SmartClustering()
        
        cluster.prepare()
        
        let textualItems = [
            TextualItem(id: uuids[0], tabId: UUID(), url: "https://www.google.com/search?q=mozart", title: "mozart - Google Search", type: TextualItemType.page),
            TextualItem(id: uuids[1], tabId: UUID(), url: "https://www.google.com/search?q=classical%20music%20mozart", title: "classical music mozart - Google Search", type: TextualItemType.page),
            TextualItem(id: uuids[2], tabId: UUID(), url: "https://www.google.com/search?q=cat", title: "cat - Google Search", type: TextualItemType.page),
            TextualItem(id: uuids[3], tabId: UUID(), url: "https://www.google.com/search?q=dog", title: "dog - Google Search", type: TextualItemType.page),
            TextualItem(id: uuids[4], tabId: UUID(), url: "https://www.google.com/search?q=worm", title: "worm - Google Search", type: TextualItemType.page),
            TextualItem(id: uuids[5], tabId: UUID(), url: "https://www.google.com/search?q=snake", title: "snake - Google Search", type: TextualItemType.page),
            TextualItem(id: uuids[6], tabId: UUID(), url: "https://www.google.com/search?q=beethoven", title: "beethoven - Google Search", type: TextualItemType.page),
            TextualItem(id: uuids[7], tabId: UUID(), url: "https://www.google.com/search?q=musique%20classique", title: "musique classique - Google Search", type: TextualItemType.page)
        ]
        var clusteredPageIds: [[UUID]] = []
        
        for texualItem in textualItems {
            clusteredPageIds = try await cluster.add(textualItem: texualItem).pageGroups
        }

        clusteredPageIds = try await cluster.changeCandidate(threshold: 0.3690).pageGroups

        expect(Set(clusteredPageIds)).to(equal(Set([[uuids[0], uuids[1], uuids[6], uuids[7]], [uuids[2], uuids[3], uuids[4], uuids[5]]])))
    }

    func testMultilingualPages() async throws {
        let cluster = SmartClustering()
        
        cluster.prepare()
        
        let textualItems = [
            TextualItem(id: UUID(), tabId: UUID(), url: SamplePageContent.enFedererWiki.url, originalContent: SamplePageContent.enFedererWiki.originalContent, type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: SamplePageContent.frFedererWiki.url, originalContent: SamplePageContent.frFedererWiki.originalContent, type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: SamplePageContent.enNadalWiki.url, originalContent: SamplePageContent.enNadalWiki.originalContent, type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: SamplePageContent.frNadalWiki.url, originalContent: SamplePageContent.frNadalWiki.originalContent, type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.youtube.com", originalContent: ["All"], type: TextualItemType.page)
        ]
        var clusteredPageIds: [[UUID]] = []
        
        for textualItem in textualItems {
            clusteredPageIds = try await cluster.add(textualItem: textualItem).pageGroups
        }
        
        expect(clusteredPageIds.count).to(equal(2))
    }
    
    func testLongAndShortText() async throws {
        let cluster = SmartClustering()
        
        cluster.prepare()
                
        let textualItems = [
            TextualItem(id: uuids[0], tabId: UUID(), url: SamplePageContent.frAndroidNvidiaShield.url, title: SamplePageContent.frAndroidNvidiaShield.title, originalContent: SamplePageContent.frAndroidNvidiaShield.originalContent, type: TextualItemType.page),
            TextualItem(id: uuids[1], tabId: UUID(), url: SamplePageContent.frAndroidSoldes.url, title: SamplePageContent.frAndroidSoldes.title, originalContent: SamplePageContent.frAndroidSoldes.originalContent, type: TextualItemType.page),
            TextualItem(id: uuids[2], tabId: UUID(), url: "https://www.google.com/search?q=bloodborne%20sculpts", title: "Bloodborne Sculpts", type: TextualItemType.page),
            TextualItem(id: uuids[3], tabId: UUID(), url: "https://www.etsy.com/market/bloodborne_sculpture?ref=seller_tag_bottom_text-1", title:"Bloodborne Sculpture | Etsy France", type: TextualItemType.page),
            TextualItem(id: uuids[4], tabId: UUID(), url: SamplePageContent.nvidiaShield4k.url, title: SamplePageContent.nvidiaShield4k.title, originalContent: SamplePageContent.nvidiaShield4k.originalContent, type: TextualItemType.page),
            TextualItem(id: uuids[5], tabId: UUID(), url: SamplePageContent.bloodborne.url, title: SamplePageContent.bloodborne.title, originalContent: SamplePageContent.bloodborne.originalContent, type: TextualItemType.page)
        ]
        var clusteredPageIds: [[UUID]] = []
        
        for textualItem in textualItems {
            clusteredPageIds = try await cluster.add(textualItem: textualItem).pageGroups
        }
        
        expect(Set(clusteredPageIds)).to(equal(Set([[uuids[2], uuids[3], uuids[5]], [uuids[0], uuids[1], uuids[4]]])))
    }
    
    func testFullemptyPages() async throws {
        let cluster = SmartClustering()
        
        cluster.prepare()
        
        let textualItems = [
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=mozart", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=classical%20music%20mozart", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=cat", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=dog", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=worm", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=snake", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=beethoven", type: TextualItemType.page),
            TextualItem(id: UUID(), tabId: UUID(), url: "https://www.google.com/search?q=musique%20classique", type: TextualItemType.page)
        ]
        var clusteredPageIds: [[UUID]] = []
        
        for textualItem in textualItems {
            clusteredPageIds = try await cluster.add(textualItem: textualItem).pageGroups
        }
        
        expect(clusteredPageIds.count).to(equal(1))
    }
    
    func testMixPageNote() async throws {
        let cluster = SmartClustering()
        
        cluster.prepare()
        
        let textualItems = [
            TextualItem(id: uuids[0], tabId: UUID(), url: SamplePageContent.enFedererWiki.url, originalContent: SamplePageContent.enFedererWiki.originalContent, type: TextualItemType.page),
            TextualItem(id: uuids[1], tabId: UUID(), url: SamplePageContent.enNadalWiki.url, originalContent: SamplePageContent.enNadalWiki.originalContent, type: TextualItemType.page),
            TextualItem(id: uuids[2], tabId: UUID(), url: "https://www.youtube.com", originalContent: ["All"], type: TextualItemType.page),
            TextualItem(id: uuids[3], tabId: UUID(), title: SamplePageContent.frFedererWiki.title, originalContent: SamplePageContent.frFedererWiki.originalContent, type: TextualItemType.note),
            TextualItem(id: uuids[4], tabId: UUID(), title: SamplePageContent.frNadalWiki.title, originalContent: SamplePageContent.frNadalWiki.originalContent, type: TextualItemType.note)
        ]
        var clusteredPageIds: [[UUID]] = []
        var clusteredNoteIds: [[UUID]] = []
        
        for textualItem in textualItems {
            (clusteredPageIds, clusteredNoteIds, _) = try await cluster.add(textualItem: textualItem)
        }
        
        expect(Set(clusteredPageIds)).to(equal(Set([[uuids[0], uuids[1]], [uuids[2]]])))
        expect(Set(clusteredNoteIds)).to(equal(Set([[uuids[3], uuids[4]], []])))
    }
    
    func testRemoveTextualItem() async throws {
        let cluster = SmartClustering()
        
        cluster.prepare()
        
        var pageUUIDs: [UUID] = []
        var tabUUIDs: [UUID] = []
        var clusteredPageIds: [[UUID]] = []
        
        for i in 0...2 {
            pageUUIDs.append(UUID())
            tabUUIDs.append(UUID())
            
            let myPage = TextualItem(id: pageUUIDs[i], tabId: tabUUIDs[i], url: "http://note.com", title: "My note", originalContent: ["This is my note"], type: TextualItemType.page)
            
            clusteredPageIds = try await cluster.add(textualItem: myPage).pageGroups
        }
        
        expect(clusteredPageIds[0].count) == 3
        expect(clusteredPageIds[0][0]) == pageUUIDs[0]
        expect(clusteredPageIds[0][1]) == pageUUIDs[1]
        expect(clusteredPageIds[0][2]) == pageUUIDs[2]
        
        clusteredPageIds = try await cluster.removeTextualItem(textualItemUUID: pageUUIDs[0], textualItemTabId: tabUUIDs[0]).pageGroups
        
        expect(clusteredPageIds[0].count) == 2
        expect(clusteredPageIds[0][0]) == pageUUIDs[1]
        expect(clusteredPageIds[0][1]) == pageUUIDs[2]
    }
    
    func testAddMultipleTimesTheSamePage() async throws {
        let cluster = SmartClustering()
        
        cluster.prepare()
        
        let pageUUID = UUID()
        let tabUUID = UUID()
        var clusteredPageIds: [[UUID]] = []
        
        for _ in 0...2 {
            let myPage = TextualItem(id: pageUUID, tabId: tabUUID, url: "http://note.com", title: "My note", originalContent: ["This is my note"], type: TextualItemType.page)
            
            clusteredPageIds = try await cluster.add(textualItem: myPage).pageGroups
        }
        
        expect(clusteredPageIds[0].count) == 1
    }
    
    func testAddMultipleTimesTheSamePageFromDifferentTab() async throws {
        let cluster = SmartClustering()
        
        cluster.prepare()
        
        let pageUUID = UUID()
        var tabUUIDs: [UUID] = []
        var clusteredPageIds: [[UUID]] = []
        
        for i in 0...2 {
            tabUUIDs.append(UUID())
            
            let myPage = TextualItem(id: pageUUID, tabId: tabUUIDs[i], url: "http://note.com", title: "My note", originalContent: ["This is my note"], type: TextualItemType.page)
            
            clusteredPageIds = try await cluster.add(textualItem: myPage).pageGroups
        }

        expect(clusteredPageIds[0].count) == 3
        expect(clusteredPageIds[0][0]) == pageUUID
        expect(clusteredPageIds[0][1]) == pageUUID
        expect(clusteredPageIds[0][2]) == pageUUID
        
        clusteredPageIds = try await cluster.removeTextualItem(textualItemUUID: pageUUID, textualItemTabId: tabUUIDs[0]).pageGroups
        
        expect(clusteredPageIds[0].count) == 2
        expect(clusteredPageIds[0][0]) == pageUUID
        expect(clusteredPageIds[0][1]) == pageUUID
    }
}

let uuids = [
    UUID(uuidString: "45EC01A0-E942-4C31-AFFB-7B69959F078A")!,
    UUID(uuidString: "8F9F9FDA-9397-4601-A576-B1E90C089A0E")!,
    UUID(uuidString: "82F81E03-C0BF-4B75-89A9-E683F2B18F21")!,
    UUID(uuidString: "55D3D2CF-ABE8-4E07-B411-70E951292E43")!,
    UUID(uuidString: "B354E048-FDA9-42F1-997A-9C7BE5BF7C9B")!,
    UUID(uuidString: "C5A7C8FC-C5AC-43A0-98A9-3B5ED3DE271F")!,
    UUID(uuidString: "B26A31C8-B700-4BF8-A084-47BD17CF5AD4")!,
    UUID(uuidString: "85E79181-6F55-4D84-A3EE-DD936EAFFCF4")!,
]

private struct SamplePageContent {
    var url: String
    var title: String
    var originalContent: [String]
}

extension SamplePageContent {
    static let frAndroidNvidiaShield = SamplePageContent(url: "https://www.amazon.fr/Nvidia-SHIELD-Support-Vendu-S%C3%A9par%C3%A9ment/dp/B07Z6RD4M9/ref=sr_1_1?keywords=shield+tv+pro+2022&qid=1649420635&sprefix=%2Caps%2C47&sr=8-1",
                                                         title: "Nvidia Shield Tv Pro (Support Shield Vendu Separement) : Amazon.Fr: High-Tech",
                                                         originalContent: ["Je possédais la version précédente (2017) et c’est naturellement que je me suis orienté vers la 2019 qui est plus rapide grâce à son processeur Nvidia Tegra X1+. L’expérience 4K HDR est au top que ce soit pour regarder des films, des séries ou encore jouer à des jeux. Cette version pro est compatible Dolby Vision HDR, Dolby Audio et HDR10 ce qui est un réel plus. Il est important de préciser que la version Pro dispose de 1 Go de RAM supplémentaire que la version classique et elle dispose aussi de 16 Go de stockage interne pour 8 Go pour la version classique.Points forts :- Design- Fluidité- Facile à configurer- Simplicité de l’interface- Capable de lire n'importe quel fichier vidéo ou de musique- Lecture multimédia au top- Le Bluetooth est utilisable pour connecter n'importe quel périphérique bluetooth- Reconnaissance vocale de qualité- Gestion HDR- 3 Go de mémoire vive et de 16 Go d'espace de stockage- 2 ports USB 3.0- Télécommande aux touches rétroéclairées, télécommande que l’on peut transformer en télécommande universelle pour contrôler le téléviseur. Accès direct à Netflix. Elle est compatible avec les versions précédentes. Il y a aussi une fonction de localisation à distance ce qui est bien pratique.- Bon Système d'upscaling- Google Assistant et Chromecast 4K intégré qui permet d’envoyer très facilement du contenu du téléphone mobile vers la télévision.Points faibles :- Pas de port MicroSD- Absence d’alimentation USB- Prix- Alimentation intégrée- Les 2 ports USB 3.0 sont trop proches l’un de l’autre, pas toujours pratique en fonction de ce que l’on branche. Il faudra donc brancher un Hub.- Play Store réduit.Ne mérite pas 5 étoiles car vu le prix, Nvidia aurait quand même pu proposer plus d'évolutions à son boitier : alimentation USB, wifi 6, plus d'espace de stockage que les 16Go, port MicroSD... J'ai achete sur la base d'articles annoncant la nouvelle version\"\"plus puissante, etc..\"\" Je cherchais une alternative a mon player Popcorn, qui puisse supporte le 4k,HDR, x265 et un meilleur audio. Mon utilisation principale est d'exploiter une base considerable de films et de CD. Je ne dirai rien de la plateforme de jeux, ca n'est pas mon truc, (sauf que je confirme que l'acceleration de perf avec le changement de modele 2019(et les 100 EUR) nes'applique qu'au Gaming et pas du tout a l'environnement android) Mes films sont organises rationellement, sur mon NAS, geres par un Librarian ecrit par moi meme sur mon PC, et executable par le PopCorn. La musique est geree par lePopCorn, plutot bien, mais le rendu audible est perfectible. Donc j'ai opte pour ce monde Android que je croyais connaitre et pour lequel j'ai programme en un temps.. mais ce que je n'avais pas anticipe c'est que Android TV , du moins tel qu'il est implemente sur le Shield, est une camisole de force contraignante, penible et pour tout dire CH..NTE au max! Apres avoir gache deux Week ends sur Internet et You Tube pour tenter de deserrer les contraintes j'hesite encore a retourner ma Shield.. Ca commence par un constat objectif: J'ai une machine pseudo-android qui utilise un espace 3840x2160 pixels au travers d'une interface simpliste digne d'un commodore avec ses gros boutons non redimensionnables, une souris absente, donc un appareil infirme au niveau de l'interface utilisateur! DRAMATIQUE! Ca continue quand on decouvre que l'on a pas droit non plus a un Browser internet decent! Pas de Chrome sur google et les substituts sont PATHETIQUES, mon troisieme Cellphone a 80 Euros fait nettement mieux! Pas question d'installer les applis de l' Android Store, on accede a des sous-store orientees perception d'abonnement et de droits, C'est la frustation TOTALE. Le support et la doc online sont inacessibles depuis la machine, donc on lit sur son tel avec chrome ou opera ou dolphin ou ou ou..la miserable douzaine de pages fournies par Nvidia.. Pas de file Manager non plus sauf a chercher dans les coins.. Pas non plus de gestionnaire d'acces NAS au niveau du Shield, cela est remis a chaque application avec des inconsequences selon que l'on parle via Kodi, ou Plex, ou VLC Et puis on se decide a tenter de Hacker pour faire sauter ces limitations frustrantes, castratrices et de fait INADMISSIBLES. Alors on se lance dans la centaine de video sur You Tube decrivant des work-around pour ameliorer la situation. La plupart sont inutiles, s'appliquent a la version precedente de Shield. Et il me semble bien qu'une des raisons de cette nouvelle version est de limiter le hacking car bientot les applis nouvellement installees via une appstore nonofficielle sont flagees par le Shield.. Bref je suis nul, sansdoute, et pas content, surement! . Car je me sens encore PLUS FRUSTRE que quand APPLE m'enferme dans ses systemes, car c'est MIEUX FAIT chez la Pomme! Pour que j'en sois a dire cela c'est vraiment que la situation est penible. Doncsi c'est pour jouer, je suis incompetent. si c'est pour regarder netflix ou myCanal (!) ca colle sans doute Si c'est pour installer un vrai Media center ca n'ira que si vous etes gentil, discipline pas exigeant et la tete bien courbee pour obeir a google TV, interface et OS de m..."])

    static let frAndroidSoldes = SamplePageContent(url: "https://www.frandroid.com/bons-plans/1216689_la-nvidia-shield-tv-pro-est-de-retour-en-promotion-juste-avant-la-fin-des-soldes",
                                                   title: "L'excellente Nvidia Shield Tv Pro Baisse Son Prix Pour La Fin Des Soldes",
                                                   originalContent: ["La Nvidia Shield TV Pro continue de trôner au sommet des box Android TV pour plusieurs raisons. Tout d’abord, c’est un boîtier multimédia complet au niveau du divertissement, que ce soit le streaming avec les applications phares telles que Netflix et autres ou encore le jeu vidéo, notamment via le service GeForce Now accessible en exclusivité sur la machine. Ensuite, c’est une box qui est encore mis à jour à l’heure actuelle, puisqu’elle tourne maintenant sous Android 11. Bref, c’est un must-have, surtout lorsqu’une promotion pointe le bout de son nez. Pour jouer, la Nvidia Shield TV Pro (2019) est une plateforme idéale afin de profiter de l’ensemble des jeux disponibles sur le Play Store. Sa puce Tegra X1+ épaulée par 3 Go de mémoire vive est 25 % plus puissante par rapport à la configuration de l’ancien modèle, pour une expérience gaming encore plus agréable. Pour aller plus loin avec des graphismes toujours plus beaux, notez qu’il est également possible d’accéder (gratuitement ou moyennant un abonnement) au service de cloud gaming GeForce Now, d’autant plus que la box est parfaitement optimisée pour. Pour améliorer votre débit, vous pourrez évidemment brancher un câble Ethernet et il y a même plusieurs ports USB pour connecter des manettes si toutes ne sont pas compatibles Bluetooth. En ce qui concerne les films et séries, la box de Nvidia propose les certifications Dolby Vision et Dolby Atmos, avec également une compatibilité 4K grâce à un nouvel algorithme plus poussé qu’auparavant, dans le but d’obtenir une meilleure qualité d’images lorsque vous visionnerez du contenu. Vous pourrez alors en profiter sur les différentes applications de streaming, mais aussi avec vos fichiers compatibles via le port USB de la Shield TV Pro, via un disque dur externe ou autre. On peut donc s’en servir comme serveur Plex en Full HD. Les données transmises par le biais de ce formulaire sont destinées à HUMANOID, société éditrice du site Frandroid en sa qualité de responsable de traitement. Elles ne seront en aucun cas cédées à des tiers. Ces données sont traitées sous réserve d'obtention de votre consentement pour vous envoyer par e-mail des actualités et informations relatives aux contenus éditoriaux publiés sur Frandroid. Vous pouvez vous opposer à tout moment à ces e-mails en cliquant sur les liens de désinscriptions présents dans chacun d'eux. Pour plus d'informations, vous pouvez consulter l'intégralité de notre politique de traitement de vos données personnelles. Vous disposez d'un droit d'accès, de rectification, d'effacement, de limitation, de portabilité et d'opposition pour motif légitime aux données personnelles vous concernant. Pour exercer l'un de ces droits, merci d'effectuer votre demande via notre formulaire de demandes d'exercices de droits dédié."])

    static let nvidiaShield4k = SamplePageContent(url: "https://www.nvidia.com/en-us/shield/",
                                                  title: "Shield Tv 4K Hdr",
                                                  originalContent: ["SHIELD TV is compact, stealth, and designed to disappear behind your entertainment center, right along with your cables. SHIELD Pro takes this performance to the next level. It’s built for the most demanding users and beautifully designed to be the perfect centerpiece of your entertainment center. Search for movies and shows, access media playback controls, change the temperature, dim the lights, and so much more. Ask questions and see the answers on your TV, including Google Photos, your calendar, sports scores, and more. Smart devices such as thermostats, doorbells, cameras, even coffee makers, can be controlled with your voice and set on schedules to make your life easier†‡. Even automate and control your smart devices‡. The Google Assistant is always on, ready to help, and makes life in your living room that much more relaxing*†."])

    static let bloodborne = SamplePageContent(url: "https://en.wikipedia.org/wiki/Bloodborne",
                                              title: "Bloodborne",
                                              originalContent: ["Bloodborne[b] is a 2015 action role-playing game developed by FromSoftware and published by Sony Computer Entertainment for the PlayStation 4. Bloodborne follows the player's character, a Hunter, through the decrepit Gothic, Victorian-era–inspired city of Yharnam, whose inhabitants are afflicted with a blood-borne disease. Attempting to find the source of the plague, the player's character unravels the city's mysteries while fighting beasts and cosmic beings. The game is played from a third-person perspective. Players control a customizable protagonist, and the gameplay is focused on strategic weapons-based combat and exploration. Players battle varied enemies while using items such as swords and firearms, exploring different locations, interacting with non-player characters, and unravelling the city's mysteries. Bloodborne began development in 2012 under the working title of Project Beast. Bearing many similarities to the Souls series of games by the same developer and director, Bloodborne was inspired by the literary works of authors H. P. Lovecraft and Bram Stoker and the architectural design of real-world locations in countries such as Romania, the Czech Republic and Scotland. Bloodborne was released in March 2015 and received critical acclaim, with praise directed at its gameplay, particularly its high difficulty level, atmosphere, sound design, Lovecraftian themes, and interconnected world design. Some criticism was directed at its technical performance at launch, which was improved with post-release updates. An expansion adding additional content, The Old Hunters, was released in November 2015. By the end of 2015, the game had sold over two million copies worldwide. Bloodborne won several awards and has been cited as one of the greatest video games ever made. Some related media and adaptations have also been released, including a card game, board game and comic book series. Bloodborne is an action role-playing game played from a third-person perspective and features elements similar to those found in the Souls series of games, particularly Demon's Souls and Dark Souls.[1] The player makes their way through different locations within the decrepit Gothic world of Yharnam while battling varied enemies, including bosses,[2] collecting different types of items that have many uses, interacting with non-player characters,[3] opening up shortcuts, and continuing through the main story. At the beginning of the game, the player creates their character, the Hunter. The player determines the basic details of the Hunter; gender, hairstyle, name, skin colour, body shape, voice, and eye colour are some of the options the player can customize. The player also chooses a starting class, known as an \"\"Origin\"\", which provides a basic backstory for the Hunter and sets the player's starting attributes. The origins, while describing the player character's past, do not have any effect on gameplay beyond altering starting stats.[4][5] Another way the player defines their Hunter is by choosing what brotherhood they are a member of. These religious societies, known as \"\"Covenants\"\", each have their views on the world of Yharnam.[6][7] The player can return to the safe zone, known as the \"\"Hunter's Dream\"\", by interacting with lanterns spread throughout the world of Yharnam. Doing so replenishes health, but repopulates all enemies in the game world, with the exception of bosses and mini-bosses. Lanterns also serve as the game's checkpoints; the player will return to the last activated lantern when they die. Positioned separate from Yharnam, the Hunter's Dream delivers some of the game's basic features to the player. Players may purchase helpful items, such as weapons, clothing and consumables, from the Messengers using Blood Echoes or Insight, level up their character by talking to the Doll, or upgrade their weapons in the workshop, among other things. Unlike Yharnam and all other locations in the game, the Hunter's Dream is considered completely safe as it is the only location in the game not to feature enemies. However, the last two boss battles of the game take place in the Hunter's Dream, although both are optional to the player.[8][9][10]"])

    static let frFedererWiki = SamplePageContent(url: "https://fr.wikipedia.org/wiki/Roger_Federer",
                                                 title: "Roger Federer",
                                                 originalContent: ["Sa victoire à Roland-Garros en 2009 lui a permis d'accomplir le Grand Chelem en carrière sur quatre surfaces différentes. En s'adjugeant ensuite l'Open d'Australie en 2010, il devient le premier joueur de l'histoire à avoir conquis l'ensemble de ses titres du Grand Chelem sur un total de cinq surfaces, depuis le remplacement du Rebound Ace australien par une nouvelle surface : le Plexicushion. Federer a réalisé le Petit Chelem de tennis à trois reprises, en 2004, 2006 et 2007, ce qui constitue à égalité avec Novak Djokovic, le record masculin toutes périodes confondues. Il est ainsi l'unique athlète à avoir gagné trois des quatre tournois du Grand Chelem deux années successives. Il atteint à trois reprises, et dans la même saison, les finales des quatre tournois majeurs, en 2006, 2007 et 2009, un fait unique dans l'histoire de ce sport."])

    static let enFedererWiki = SamplePageContent(url: "https://en.wikipedia.org/wiki/Roger_Federer",
                                                 title: "",
                                                 originalContent: ["Federer has played in an era where he dominated men's tennis together with Rafael Nadal and Novak Djokovic, who have been collectively referred to as the Big Three and are widely considered three of the greatest tennis players of all-time. A Wimbledon junior champion in 1998, Federer won his first Grand Slam singles title at Wimbledon in 2003 at age 21. In 2004, he won three out of the four major singles titles and the ATP Finals, a feat he repeated in 2006 and 2007. From 2005 to 2010, Federer made 18 out of 19 major singles finals. During this span, he won his fifth consecutive titles at both Wimbledon and the US Open. He completed the career Grand Slam at the 2009 French Open after three previous runner-ups to Nadal, his main rival up until 2010. At age 27, he also surpassed Pete Sampras's then-record of 14 Grand Slam men's singles titles at Wimbledon in 2009."])

    static let enNadalWiki = SamplePageContent(url: "https://en.wikipedia.org/wiki/Rafael_Nadal",
                                               title: "",
                                               originalContent: ["From childhood through most of his professional career, Nadal was coached by his uncle Toni. He was one of the most successful teenagers in ATP Tour history, reaching No. 2 in the world and winning 16 titles before his 20th birthday, including his first French Open and six Masters events. Nadal became No. 1 for the first time in 2008 after his first major victory off clay against his rival, the longtime top-ranked Federer, in a historic Wimbledon final. He also won an Olympic gold medal in singles that year in Beijing. After defeating Djokovic in the 2010 US Open final, the 24-year-old Nadal became the youngest man in the Open Era to achieve the career Grand Slam, and also became the first man to win three majors on three different surfaces (hard, grass and clay) the same calendar year. With his Olympic gold medal, he is also one of only two male players to complete the career Golden Slam."])

    static let frNadalWiki = SamplePageContent(url: "https://fr.wikipedia.org/wiki/Rafael_Nadal",
                                               title: "Rafael Nadal",
                                               originalContent: ["Il est considéré par tous les spécialistes comme le meilleur joueur sur terre battue de l'histoire du tennis, établissant en effet des records majeurs, et par la plupart d'entre eux comme l'un des meilleurs joueurs de simple de tous les temps, si ce n’est le meilleur. Il a remporté vingt tournois du Grand Chelem (un record qu'il détient avec Roger Federer et Novak Djokovic) et est le seul joueur à avoir remporté treize titres en simple dans un de ces quatre tournois majeurs : à Roland-Garros où il s'est imposé de 2005 à 2008, de 2010 à 2014, puis de 2017 à 2020. À l'issue de l'édition 2021, où il est détrôné en demi-finale par Novak Djokovic, il présente un bilan record de cent-cinq victoires pour trois défaites dans ce tournoi parisien, et ne compte aucune défaite en finale. Il a remporté également le tournoi de Wimbledon en 2008 et 2010, l'Open d'Australie 2009 et l'US Open 2010, 2013, 2017 et 2019. Il est ainsi le septième joueur de l'histoire du tennis à réaliser le « Grand Chelem en carrière » en simple. À ce titre, Rafael Nadal est le troisième joueur et le plus jeune à s'être imposé durant l'ère Open dans les quatre tournois majeurs sur quatre surfaces différentes, performance que seuls Roger Federer, Andre Agassi et Novak Djokovic ont accomplie."])
}
