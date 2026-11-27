# Datasets e créditos

O AtlasLeaf **não coletou a maior parte das imagens de treino** — ele combina datasets públicos de
doenças de soja. Esta página dá o crédito devido a cada fonte e registra licença e uso. É requisito de
licença (algumas fontes são CC BY, que **exige atribuição**) e de honestidade científica.

## Resumo: o que o modelo publicado usa

**O modelo `field7` (`v2.0`) é treinado com ASDID (câmera Canon) + Mendeley V2** nas 7 classes de campo,
mantendo as fotos **Motorola** do ASDID como holdout exclusivo (`splits_camera_mendeley_train.json`):

| Conjunto | Imagens | Fontes |
|---|---:|---|
| train | 5.936 | `asdid` (Canon) + `mendeleyv2` |
| val | 1.040 | `asdid` (Canon) + `mendeleyv2` |
| test (cross-camera) | 2.477 | só `asdid` (Motorola — nunca vista no treino) |

Contagem por fonte no dataset reconstruído (`datasets/unified_resized/manifest.json`, 10.147 imagens — pasta
**não versionada**):

| Fonte (`source`) | Imagens | Usada no field7 v2.0? |
|---|---:|---|
| `asdid` | 9.648 | **Sim** — base principal (Canon no treino/val, Moto no holdout) |
| `mendeleyv2` | 499 | **Sim** — 295 imgs nas classes `healthy`, `asian_rust` e `cercospora_blight` |

---

## Fontes usadas pelo modelo

### ASDID — Auburn Soybean Disease Image Dataset

Fotos de campo de folhas de soja de estações de pesquisa do Alabama (EUA), câmeras Canon EOS 7D
Mark II e smartphone Motorola Moto Z2 Play; ~9.648 imagens em 8 categorias (sadia, crestamento
bacteriano, cercospora, míldio, mancha olho-de-rã, ferrugem, mancha alvo, deficiência de potássio).

- **Citação:** Bevers, N., Sikora, E. J., & Hardy, N. B. (2022). *Pictures of diseased soybean leaves
  by category captured in field and with controlled backgrounds: Auburn soybean disease image dataset
  (ASDID)* [Dataset]. Dryad. https://doi.org/10.5061/dryad.41ns1rnj3
- **Artigo associado:** Bevers, N., Sikora, E. J., & Hardy, N. B. (2022). *Soybean disease
  identification using original field images and transfer learning with convolutional neural
  networks.* Computers and Electronics in Agriculture, 203, 107449.
- **Licença:** CC0 1.0 (Dryad — domínio público). Atribuição não é exigida, mas é dada por cortesia.

### Mendeley V2 — Multi-Class Soybean Leaf Disease Dataset (2ª fonte de campo)

Fotos de campo (DSLR/smartphone), 5 classes (`Healthy`, `Bacterial Blight`, `Cercospora Leaf Blight`,
`Sudden Death Syndrome`, `Rust`; 499 imagens no total, sendo 295 nas 3 classes compatíveis com `field7`).
Integrada na `v2.0` como 2ª fonte independente para reforçar *Cercospora* e combater o viés de fonte.

- **Citação:** Thorwat, M., Magdum, P., Jadhav, S., Sutar, A., & Oswal, R. (2026). *Multi-Class Soybean
  Leaf Disease Dataset: Healthy and Diseased Leaf Images for Machine Learning* (V2) [Dataset]. Mendeley
  Data. https://data.mendeley.com/datasets/6fhphxg297/2
- **Licença:** **CC BY 4.0** — atribuição obrigatória cumprida aqui e no `README.md`.

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
