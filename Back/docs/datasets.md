# Datasets e créditos

O AtlasLeaf **não coletou a maior parte das imagens de treino** — ele combina datasets públicos de
doenças de soja. Esta página dá o crédito devido a cada fonte e registra licença e uso. É requisito de
licença (algumas fontes são CC BY, que **exige atribuição**) e de honestidade científica.

## Resumo: o que o modelo publicado usa

**O modelo field7 é treinado exclusivamente com o ASDID.** O split por câmera
(`splits_camera.json`) divide por câmera (Canon vs Motorola), rótulo que só o ASDID tem — logo as
outras fontes ficam de fora automaticamente. Composição real do treino/val/teste do field7:

| Conjunto | Imagens | Fontes |
|---|---:|---|
| train | 5.854 | só `asdid` |
| val | 1.028 | só `asdid` |
| test (cross-camera) | 2.655 | só `asdid` |

Contagem por fonte no dataset unificado (`datasets/unified/manifest.json`, 10.730 imagens — pasta
**não versionada**, resíduo da unificação v3.0/3.1):

| Fonte (`source`) | Imagens | Usada no field7? |
|---|---:|---|
| `asdid` | 10.027 | **Sim** — é a única base do modelo |
| `kaggle_soybean` | 618 | Não — classes fora do field7 e sem rótulo de câmera |
| `soynet` | 85 | Não — idem |

---

## Fonte usada pelo modelo

### ASDID — Auburn Soybean Disease Image Dataset (única base do field7)

Fotos de campo de folhas de soja de estações de pesquisa do Alabama (EUA), câmeras Canon EOS 7D
Mark II e smartphone Motorola Moto Z2 Play; ~9.981 imagens em 8 categorias (sadia, crestamento
bacteriano, cercospora, míldio, mancha olho-de-rã, ferrugem, mancha alvo, deficiência de potássio).

- **Citação:** Bevers, N., Sikora, E. J., & Hardy, N. B. (2022). *Pictures of diseased soybean leaves
  by category captured in field and with controlled backgrounds: Auburn soybean disease image dataset
  (ASDID)* [Dataset]. Dryad. https://doi.org/10.5061/dryad.41ns1rnj3
- **Artigo associado:** Bevers, N., Sikora, E. J., & Hardy, N. B. (2022). *Soybean disease
  identification using original field images and transfer learning with convolutional neural
  networks.* Computers and Electronics in Agriculture, 203, 107449.
- **Licença:** CC0 1.0 (Dryad — domínio público). Atribuição não é exigida, mas é dada por cortesia.

---

## Presentes no dataset unificado, mas NÃO usadas pelo field7

Estas fontes vieram da unificação v3.0/3.1 e continuam no `datasets/unified/` (não versionado), porém
**o modelo publicado não as utiliza** (ver resumo acima). Documentadas por completude; se um treino
futuro passar a usá-las, mover para a seção anterior e resolver as pendências de licença.

### SoyNet — High-resolution Indian soybean image dataset (85 imgs, não usadas)

Dataset indiano de imagens de folha de soja em estúdio/fundo controlado.

- **Citação:** Rajput, A. S., Shukla, S., & Thakur, S. S. (2023). *SoyNet: A high-resolution Indian
  soybean image dataset for leaf disease classification.* Data in Brief, 49, 109447.
  https://doi.org/10.1016/j.dib.2023.109447
- **Licença:** verificar na fonte se for usar (Data in Brief / Mendeley — tipicamente CC BY 4.0).

### Kaggle (`kaggle_soybean`) — 618 imgs, não usadas · procedência não registrada

618 imagens de origem Kaggle no manifest, sem URL/autor. Como o field7 **não as usa** e a pasta não é
versionada, não há pendência de conformidade hoje. Se um dia forem usadas, é preciso rastrear o
dataset exato no Kaggle (autor + licença) — muitos "soybean disease" do Kaggle são derivados de
PlantVillage/PlantDoc.

---

## Fontes planejadas (ainda não integradas)

### Mendeley — Multi-Class Soybean Leaf Disease Dataset (candidata a 2ª fonte de campo)

Fotos de campo (DSLR/smartphone), 5 classes (Healthy, Bacterial Blight, Cercospora Leaf Blight,
Sudden Death Syndrome, Soybean Rust). Candidata a 2ª fonte independente para combater o confounding de
fonte (ver README, seção de viés de fonte).

- **Citação:** Thorwat, M., Magdum, P., Jadhav, S., Sutar, A., & Oswal, R. (2026). *Multi-Class Soybean
  Leaf Disease Dataset: Healthy and Diseased Leaf Images for Machine Learning* (V2) [Dataset]. Mendeley
  Data. https://data.mendeley.com/datasets/6fhphxg297/2
- **Licença:** CC BY 4.0 — **exige atribuição** ao usar. Mover para a tabela "Fontes usadas" quando
  integrada.

### Coleta própria — oeste baiano

Dataset de campo a ser coletado na região do usuário (ver `docs/protocolo_coleta.md`). Fonte primária
própria; sem exigência de crédito externo.

---

## BibTeX

```bibtex
@dataset{bevers2022asdid,
  author    = {Bevers, Noah and Sikora, Edward J. and Hardy, Nate B.},
  title     = {Pictures of diseased soybean leaves by category captured in field and with
               controlled backgrounds: Auburn soybean disease image dataset (ASDID)},
  year      = {2022},
  publisher = {Dryad},
  doi       = {10.5061/dryad.41ns1rnj3}
}

@article{bevers2022soybean,
  author  = {Bevers, Noah and Sikora, Edward J. and Hardy, Nate B.},
  title   = {Soybean disease identification using original field images and transfer learning
             with convolutional neural networks},
  journal = {Computers and Electronics in Agriculture},
  volume  = {203},
  pages   = {107449},
  year    = {2022}
}

@article{rajput2023soynet,
  author  = {Rajput, Amit Singh and Shukla, Shalini and Thakur, Sanjeev Singh},
  title   = {SoyNet: A high-resolution Indian soybean image dataset for leaf disease classification},
  journal = {Data in Brief},
  volume  = {49},
  pages   = {109447},
  year    = {2023},
  doi     = {10.1016/j.dib.2023.109447}
}

@dataset{thorwat2026soybean,
  author    = {Thorwat, Madhuri and Magdum, Pranali and Jadhav, Shweta and Sutar, Anushka
               and Oswal, Riya},
  title     = {Multi-Class Soybean Leaf Disease Dataset: Healthy and Diseased Leaf Images
               for Machine Learning},
  year      = {2026},
  version   = {V2},
  publisher = {Mendeley Data},
  doi       = {10.17632/6fhphxg297.2}
}
```
