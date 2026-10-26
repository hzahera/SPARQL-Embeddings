// Numbers reported in the paper (current manuscript tables). Edit here if the tables change.
// Encoder order is the same in every array.
window.PAPER = {
  encoders: ["MPNet", "E5-base-v2", "BGE-base", "GTE-base", "Qwen3-Embedding",
             "Gemma-3-270M", "CodeBERT", "GraphCodeBERT", "Qwen2.5-Coder", "SPBERT"],
  families: ["text", "text", "text", "text", "text", "general", "code", "code", "code", "sparql"],

  // Table "Retrieval of high-overlap queries" (stored labels), Hits@1.
  // zs + gain come from Tables/compact/gain_*.dat (fine-tuned = mean over five seeds).
  retrievalStored: {
    dbpedia: {
      zs:   [0.368, 0.406, 0.348, 0.331, 0.402, 0.310, 0.338, 0.314, 0.269, 0.316],
      gain: [0.189, 0.171, 0.210, 0.232, 0.173, 0.244, 0.180, 0.190, 0.289, 0.218],
      baselines: { "Random": 0.02, "TF-IDF": 0.58, "BM25": 0.40, "Normalized match": 0.06,
                   "Algebra similarity": 0.26, "Query-form features": 0.09 }
    },
    wikidata: {
      zs:   [0.295, 0.333, 0.306, 0.279, 0.374, 0.261, 0.278, 0.316, 0.248, 0.317],
      gain: [0.141, 0.098, 0.118, 0.145, 0.069, 0.169, 0.129, 0.100, 0.183, 0.098],
      baselines: { "Random": 0.00, "TF-IDF": 0.37, "BM25": 0.37, "Normalized match": 0.04,
                   "Algebra similarity": 0.29, "Query-form features": 0.17 }
    }
  },

  // Table "Retrieval among all test queries with complete labels" (Hits@1, MRR, nDCG@10).
  retrievalComplete: {
    dbpedia: [
      ["Random ranking", 0.07, 0.14, 0.07, "base"],
      ["Query-form features", 0.23, 0.33, 0.27, "base"],
      ["Algebra similarity", 0.39, 0.50, 0.44, "base"],
      ["BM25", 0.53, 0.64, 0.57, "base"],
      ["TF-IDF", 0.58, 0.66, 0.57, "base"],
      ["Encoders, zero-shot (mean of 10)", 0.49, 0.58, 0.50, "zs"],
      ["Encoders, results-aware fine-tuned (mean of 10)", 0.70, 0.77, 0.71, "ft"]
    ],
    wikidata: [
      ["Random ranking", 0.00, 0.02, 0.01, "base"],
      ["Query-form features", 0.17, 0.26, 0.23, "base"],
      ["Algebra similarity", 0.29, 0.36, 0.34, "base"],
      ["BM25", 0.37, 0.43, 0.41, "base"],
      ["TF-IDF", 0.37, 0.41, 0.39, "base"],
      ["Encoders, zero-shot (mean of 10)", 0.30, 0.38, 0.34, "zs"],
      ["Encoders, results-aware fine-tuned (mean of 10)", 0.43, 0.49, 0.45, "ft"]
    ]
  },

  // Matched control: same encoders (E5, Qwen3-Embedding, CodeBERT), seeds and protocol,
  // trained on syntax labels instead of answer overlap. Mean Hits@1, complete labels.
  syntaxControl: {
    dbpedia:  { syntax: 0.52, answers: 0.72, diff: "0.20 [0.17, 0.23]" },
    wikidata: { syntax: 0.36, answers: 0.43, diff: "0.06 [0.05, 0.08]" }
  },

  // Wikidata test anchors split by whether their template occurs in training.
  seenUnseen: {
    seen:   { n: 417, tfidf: 0.18, ft: 0.27, gain: "0.13 [0.11, 0.16]" },
    unseen: { n: 425, tfidf: 0.55, ft: 0.58, gain: "0.12 [0.10, 0.14]" }
  },

  // Table "Approximate result caching" at theta = 0.8, K = 128.
  caching: {
    dbpedia: {
      zsAcc: [0.28, 0.15, 0.21, 0.15, 0.33, 0.15, 0.15, 0.15, 0.15, 0.15],
      ftAcc: [0.75, 0.79, 0.72, 0.71, 0.67, 0.70, 0.61, 0.61, 0.65, 0.59],
      ftCHR: [0.21, 0.19, 0.29, 0.32, 0.28, 0.26, 0.37, 0.31, 0.29, 0.35],
      ftFHR: [0.01, 0.00, 0.03, 0.04, 0.05, 0.04, 0.10, 0.07, 0.06, 0.11],
      oracleCHR: 0.72, n: 473
    },
    wikidata: {
      zsAcc: [0.06, 0.01, 0.01, 0.01, 0.10, 0.01, 0.01, 0.01, 0.01, 0.01],
      ftAcc: [0.22, 0.21, 0.22, 0.21, 0.29, 0.27, 0.21, 0.22, 0.37, 0.21],
      ftCHR: [0.14, 0.13, 0.14, 0.14, 0.08, 0.08, 0.13, 0.14, 0.06, 0.14],
      ftFHR: [0.44, 0.44, 0.46, 0.46, 0.19, 0.22, 0.45, 0.45, 0.10, 0.44],
      oracleCHR: 0.30, n: 968
    }
  },

  // Risk-coverage points (Tables/appendix/riskcov.tex): [hit rate SHR, risk = 1 - accuracy]
  // for thresholds 0.50 ... 0.99, fine-tuned, pooled over seeds and stream orders.
  thresholds: [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99],
  riskCoverage: {
    dbpedia: {
      "MPNet": [[0.8542,0.5707],[0.7488,0.4584],[0.5428,0.3426],[0.4156,0.3030],[0.2820,0.2489],[0.1625,0.1753],[0.0858,0.0365],[0.0476,0.0000],[0.0349,0.0000],[0.0300,0.0000],[0.0274,0.0000]],
      "Qwen3-Embedding": [[0.9043,0.6363],[0.8155,0.5449],[0.6614,0.4540],[0.5477,0.4161],[0.4195,0.3323],[0.2669,0.2671],[0.1293,0.1629],[0.0573,0.0192],[0.0443,0.0000],[0.0364,0.0000],[0.0294,0.0000]],
      "CodeBERT": [[0.9440,0.6976],[0.8917,0.6118],[0.8053,0.4877],[0.7353,0.4315],[0.6058,0.3881],[0.3984,0.3522],[0.1950,0.3148],[0.0787,0.0913],[0.0417,0.0162],[0.0325,0.0000],[0.0260,0.0000]]
    },
    wikidata: {
      "MPNet": [[0.8138,0.8300],[0.7392,0.8029],[0.6867,0.7873],[0.6632,0.7761],[0.6438,0.7762],[0.6125,0.7863],[0.5424,0.8056],[0.3112,0.8035],[0.2129,0.7729],[0.1618,0.7614],[0.0841,0.7436]],
      "Qwen3-Embedding": [[0.6159,0.7837],[0.4756,0.7519],[0.3586,0.7192],[0.3196,0.7161],[0.2752,0.7060],[0.2033,0.6763],[0.1332,0.6323],[0.0581,0.4776],[0.0321,0.3466],[0.0228,0.3158],[0.0095,0.0649]],
      "CodeBERT": [[0.8359,0.8358],[0.7474,0.8106],[0.6878,0.7961],[0.6655,0.7883],[0.6471,0.7921],[0.6192,0.7983],[0.5867,0.8232],[0.5340,0.8616],[0.5032,0.8817],[0.4778,0.9013],[0.4178,0.9207]]
    }
  },

  // RQ1 Spearman rho between cosine and J on stored held-out pairs (mean over encoders).
  alignment: { dbpedia: { zs: 0.70, ft: 0.71 }, wikidata: { zs: 0.68, ft: 0.66 } },

  dataset: {
    dbpedia:  { queries: 4727, split: "3,781 / 473 / 473", anchors: "3,737 / 469 / 468", pos: 39430, neg: 93480, j1: "79%", templates: 121, seen: "98%" },
    wikidata: { queries: 9679, split: "7,743 / 968 / 968", anchors: "6,734 / 842 / 842", pos: 37208, neg: 168360, j1: "79%", templates: 4989, seen: "50%" }
  }
};
