# Protocolo de Coleta — AtlasLeaf (oeste baiano)

Objetivo: montar um conjunto de fotos de doenças de soja do **oeste baiano** para treinar
**junto com o ASDID**. O que faz o modelo generalizar não é volume bruto — é **diversidade**
(várias fazendas, telefones, datas) e **rótulo confiável**. Este documento é pra alinhar a coleta
com o agrônomo (seu pai / seu irmão).

---

## 1. Por que isso funciona (resumo de 30s)

Hoje o modelo só viu um domínio por doença e "decora" a fonte (câmera/região/luz) em vez da lesão.
Ao adicionar fotos da **sua região**, dois ganhos: (a) a sua lavoura vira *in-domain* (o modelo
passa a acertar onde você usa), e (b) com a **mesma doença** em duas fontes (ASDID + Bahia) ele é
forçado a aprender a lesão de verdade, não a câmera. O maior salto é de 1→2 domínios.

---

## 2. Quais doenças — escopo por órgão fotografado

**Princípio: o modelo só enxerga o que está na foto. O órgão que você fotografa já é o filtro.**
Quem fotografa uma folha está dizendo "o sintoma que vejo está na folha". Por isso o escopo é
organizado por **órgão**, não por "todas as doenças da soja". Doença cujo sinal está na raiz/haste
não se identifica por foto de folha — não importa quão comum seja no talhão.

### 2a. Folha — escopo principal

**Prioridade máxima — as que o ASDID já tem (a sobreposição é o ouro):**

| Pasta (nome exato) | Doença |
|---|---|
| `healthy` | Folha sadia |
| `asian_rust` | Ferrugem asiática |
| `target_spot` | Mancha alvo |
| `cercospora_blight` | Crestamento de cercospora |
| `frogeye_leaf_spot` | Mancha olho-de-rã |
| `downy_mildew` | Míldio |
| `potassium_deficiency` | Deficiência de potássio |

> `cercospora_blight` foi o ponto fraco do modelo atual (54% de recall). **Priorize volume aqui.**

**Fracas que já existem no dataset e são comuns aqui — reforçar na coleta (fonte nova = ouro):**

| Pasta | Doença | Imagens hoje |
|---|---|---|
| `septoria_brown_spot` | Mancha parda (*Septoria glycines*) | 48 |
| `powdery_mildew` | Oídio (*Erysiphe diffusa*) | 120 |
| `sudden_death_syndrome` | SDS / fusariose foliar (*Fusarium virguliforme*) | 110 |

> SDS é doença de raiz, mas o sintoma **foliar** (clorose internerval) é característico → classe
> foliar válida. As três estão com volume baixo; reforço da Bahia move a agulha.

**Novas foliares a abrir (dão lesão na folha, não existem no dataset):**

| Pasta | Doença |
|---|---|
| `anthracnose` | Antracnose (*Colletotrichum truncatum*) |
| `ascochyta_spot` | Mancha de Ascochyta (*Ascochyta sojae*) |

### 2b. Haste/vagem — trilha nova

| Pasta | Doença |
|---|---|
| `white_mold` | Mofo-branco (*Sclerotinia sclerotiorum*) |

Mofo-branco **não aparece na folha** — vive na **haste/vagem**. Entra mesmo assim porque é **comum**
e **visualmente inconfundível** (micélio branco cotonoso + escleródios pretos). É a primeira classe de
uma trilha "fotografe a haste", com seletor de órgão no app (folha vs. haste).

> ⚠️ **Rótulo = o sinal branco na haste/vagem, NUNCA a folha murcha acima da lesão.** A murcha é
> sintoma secundário ambíguo (parece seca / deficiência). Treinar `white_mold` com folha murcha
> reintroduz exatamente a confusão que queremos evitar. Fotografe o chumaço, não a murcha.

### 2c. Fora de escopo (não fotografar pra classificar)

Podridão de carvão (*Macrophomina phaseolina*), seca da haste e da vagem (*Diaporthe phaseolorum*),
podridão de rizoctonia (*Rhizoctonia solani*). São de raiz/haste **e** na folha dão só murcha/amarelão
genérico, sem assinatura distintiva — nem folha nem órgão têm sinal limpo o bastante. O modelo só
chutaria.

### 2d. Requisito do modelo: saber dizer "não sei"

Para murcha / amarelão inespecífico (que pode ser doença de raiz fora de escopo, seca ou deficiência),
o modelo precisa de **limiar de confiança + saída "sintoma inespecífico — verifique haste e raiz"** em
vez de cravar um rótulo foliar. É o que protege contra a folha murcha ambígua.

> Para cada doença nova, crie a pasta com nome em inglês_snake_case — o script de integração cria o id
> automaticamente. Doença nova só existe na sua fonte → ajuda a **sua** região, mas só generaliza pra
> fora quando houver uma 2ª fonte. As classes que o ASDID já tem generalizam desde a 1ª foto baiana.

---

## 3. Quantas fotos

| Meta | Por doença | Observação |
|---|---|---|
| Mínimo pra ver sinal | **30–50** | já dá pra medir a tendência com as ferramentas do projeto |
| Bom | **150–300** | move a agulha de verdade |
| Ideal (difíceis) | **300+** | especialmente cercospora e as visualmente ambíguas |

**Distribua** cada doença por **≥3 fazendas** e **≥2 telefones diferentes**. 300 fotos de uma
fazenda/um telefone valem menos que 100 bem espalhadas — sem diversidade, vira "mais uma fonte estreita".

---

## 4. Como fotografar

- **Uma folha por foto**, ocupando ~60–80% do quadro, **em foco**.
- **Luz natural difusa** (evite sol a pino estourando a imagem e sombras duras). Manhã/fim de tarde ajudam.
- Fotografe a **face com a lesão**; se a doença aparece nos dois lados (ex. ferrugem no verso), inclua os dois.
- Capture **vários estágios de severidade** (início, médio, avançado) — não só o caso "livro-texto".
- Inclua bastante `healthy` da **mesma lavoura** (o modelo precisa aprender o "normal" da sua região).
- Evite: foto tremida/desfocada, várias folhas amontoadas, print de tela, zoom digital extremo, folha molhada com reflexo forte.
- **Mofo-branco (`white_mold`):** mesma regra de foco/luz, mas o alvo é a **haste/vagem com o micélio branco e os escleródios pretos** ocupando o quadro — não a folha. (ver ⚠️ na seção 2b).

## 5. Eixos de diversidade (quanto mais, melhor)

Fazenda • telefone/câmera • data (espalhe pela safra) • hora do dia • cultivar/variedade • estágio da planta (V/R).
Anote isso — vira o que permite medir "funciona numa fazenda que eu não amostrei".

---

## 6. Rótulo (aqui entra o agrônomo)

- **Todo rótulo confirmado por agrônomo.** Na dúvida entre duas doenças, **não chute**: jogue numa
  pasta `descartar/` (o script ignora). Rótulo errado no treino **derruba o teto** do modelo inteiro.
- Casos ambíguos que idealmente pedem **confirmação laboratorial**: SDS vs outras murchas,
  bacteriana vs fúngica, mancha alvo vs olho-de-rã em estágio inicial.
- Registre quem confirmou e como (visual / lab) na planilha (seção 8).

---

## 7. Estrutura de pastas (encaixa direto no pipeline)

Organize por **fazenda → doença**:

```
bahia_raw/
  fazenda_saojose/
    asian_rust/        IMG_0001.jpg ...
    target_spot/       IMG_0042.jpg ...
    healthy/           ...
  fazenda_riogrande/
    cercospora_blight/ ...
    asian_rust/        ...
  fazenda_03/
    ...
  descartar/           (rótulo incerto — ignorado)
```

Regras:
- Nome da pasta de fazenda: livre, mas **curto e sem espaço** (`fazenda_saojose`, `f03`). É o que
  permite o **split por fazenda** (treinar em umas, testar noutra).
- Nome da pasta de doença: **exatamente** os nomes da tabela da seção 2 (inglês_snake_case).
- Nome do arquivo: qualquer um (o script renomeia ao integrar). Se der, mantenha o original do celular
  (a data EXIF ajuda).
- **Não** misture telefones dentro de uma foto/pasta de forma que perca a rastreabilidade — se puder,
  registre o aparelho na planilha.

---

## 8. Planilha de acompanhamento (CSV simples)

Mantenha um `bahia_raw/registro.csv` com uma linha por lote de fotos:

```
fazenda,doenca,telefone,data,cultivar,estagio,qtd,confirmado_por,metodo
fazenda_saojose,asian_rust,galaxy_s21,2026-02-10,BRS-8383,R5,45,pai,visual
fazenda_riogrande,cercospora_blight,iphone13,2026-02-14,?,R4,30,irmao,lab
```

Isso não entra no modelo, mas é o que te deixa auditar diversidade e rastrear problema depois.

---

## 9. Erros que anulam o esforço (não faça)

- Tudo no mesmo dia, mesma fazenda, mesmo telefone.
- Rotular no "achismo" pra completar número.
- Só o caso clássico de cada doença (sem variação de severidade).
- Recortar/editar as fotos antes (mande a original; o pipeline cuida do pré-processo).

---

## 10. Assim que tiver ~30–50 por doença

Rode a integração + avaliação (ver `scripts/integrate_dataset.py`) e o
`scripts/evaluate_field.py`. Você já vê a tendência
e decide onde reforçar a coleta. Não precisa juntar tudo antes de saber se está indo bem.
