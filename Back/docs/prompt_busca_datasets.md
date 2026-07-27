# Prompt para buscar datasets (Claude / Gemini)

Cole o bloco abaixo no Claude (com busca/pesquisa web ativada) e no Gemini. Rode nos **dois** e
compare os resultados. **Verifique cada link à mão** — modelos de linguagem às vezes inventam URLs.

---

## Prompt (copiar tudo a partir daqui)

Você é um assistente especialista em **sourcing de datasets de visão computacional para agronomia**.
Preciso encontrar **datasets públicos de imagens de doenças foliares de soja (Glycine max)** para
**complementar** um dataset que já tenho, com o objetivo de melhorar a generalização de um
classificador para **condições reais de campo no oeste da Bahia, Brasil (bioma Cerrado)**.

### Contexto (importante para você filtrar bem)
- Já uso o **ASDID (Annotated Soybean Disease Image Dataset)** — ~10 mil fotos de campo, mas de
  poucas câmeras e provavelmente de uma única região. Meu problema é **falta de diversidade de
  domínio**: o modelo decora a "assinatura" da fonte (câmera, região, iluminação) em vez da lesão.
- Portanto **não quero mais imagens do mesmo domínio**. Quero fontes que **adicionem diversidade**:
  outras regiões (idealmente Brasil/tropical/Cerrado), outras câmeras/telefones, outras condições de campo.
- **Fotos de campo (in-field) são muito mais valiosas que fotos de laboratório/estúdio** (folha
  isolada em fundo branco/preto). Diga sempre qual é o caso.

### O que priorizar (doenças, por ordem de importância)
1. **Crestamento de cercospora** — *Cercospora kikuchii* (PRIORIDADE MÁXIMA, é minha classe mais fraca)
2. Ferrugem asiática — *Phakopsora pachyrhizi*
3. Mancha alvo — *Corynespora cassiicola*
4. Mancha olho-de-rã — *Cercospora sojina*
5. Míldio — *Peronospora manshurica*
6. Folha sadia (soja saudável, em campo)
7. Deficiência de potássio (sintoma nutricional)

Também tenho interesse (expansão futura) em: antracnose (*Colletotrichum truncatum*),
mofo branco (*Sclerotinia sclerotiorum*), mancha parda/septoriose (*Septoria glycines*),
mancha bacteriana (*Pseudomonas savastanoi* pv. *glycinea*), oídio (*Microsphaera diffusa*),
míldio, viroses (mosaico), síndrome da morte súbita (*Fusarium virguliforme*).

### Requisitos
- Imagens **rotuladas por doença** (classificação) ou com anotação que permita derivar o rótulo.
- **Licença permissiva e explícita** (CC0, CC-BY, CC-BY-SA, MIT, Apache, domínio público, ou
  "free for research"). Se a licença não estiver clara, diga isso.
- Acessível para download (não atrás de paywall/pedido formal), ou explique como obter.

### Onde procurar
Kaggle, Zenodo, Mendeley Data, Roboflow Universe, Hugging Face Datasets, IEEE DataPort,
Figshare, GitHub, Papers with Code, e **datasets liberados junto a artigos científicos**
(Google Scholar / Semantic Scholar). Vale muito procurar fontes **brasileiras**: Embrapa Soja,
repositórios de universidades (ESALQ/USP, UFV, UFU, UnB), e artigos de fitopatologia da soja no Cerrado.

### Como responder (para CADA dataset encontrado, uma linha de tabela)
| Nome | URL | Licença | Nº imagens | Doenças de soja cobertas | Campo ou laboratório? | Região/país | Resolução/câmera (se houver) | É derivado de outro? (PlantVillage/PlantDoc/ASDID?) | Confiança de que existe (alta/média/baixa) |

### Regras de honestidade (críticas)
- **NÃO invente URLs nem datasets.** Se não tiver certeza que existe, marque confiança "baixa" e diga.
- Muitos datasets de "soybean disease" no Kaggle são **cópias/derivados** de PlantVillage, PlantDoc ou
  uns dos outros. **Sinalize duplicatas e lineage** — não quero adicionar o mesmo domínio repetido.
- Diga explicitamente quando um dataset é **laboratório/estúdio** (menos útil para mim).
- No fim, faça um **ranking dos 3-5 melhores para o meu caso** (campo + Brasil/tropical + cercospora
  + licença ok) e explique por que, e o que ainda falta cobrir.

---

## Como usar

1. Rode nos dois (Claude com pesquisa web; Gemini). Peça a tabela completa.
2. **Abra cada URL você mesmo** antes de confiar — confira que existe, a licença e se é campo ou lab.
3. Junte os resultados dos dois, tire duplicatas, e priorize: **campo > Brasil/tropical > cercospora > licença clara**.
4. Para os candidatos bons, baixe uma amostra pequena e me traga — eu meço se o domínio realmente
   difere do ASDID (o mesmo teste de separabilidade de fonte que já rodamos) antes de você investir em usar tudo.
