# Experiments Log

매 시도마다 1줄씩 추가. 한국어 코멘트 OK.

| v   | Date       | Model                          | CV R² (mean ± std) | Public LB R² | Submission file                                  | Note (한국어 OK)                                                  |
|-----|------------|--------------------------------|--------------------|--------------|--------------------------------------------------|--------------------------------------------------------------------|
| 01  | 2026-05-30 | Linear Regression (LSM) + median imputation | -0.45721 ± 0.97914 | (안올림)     | Baek_Seunghan_v01_baseline_lsm.csv               | Baseline. 평균값 예측보다 못함. 한 fold (-2.41)가 outlier로 폭망. Lec 07 slide 11 "highly sensitive to outliers" 그대로. |
| 02  | 2026-05-30 | Ridge + PolynomialFeatures(d=2) + StandardScaler, α=1000 | -0.83883 ± 1.47252 | (안올림)     | Baek_Seunghan_v02_ridge_poly2_alpha1000.csv | Poly가 outlier를 더 증폭. degree=3은 더 나쁨. |
| 03  | 2026-05-30 | Ridge + Poly(d=1) on signed-log target, α=100 | -25.94809 ± 51.02429 | (안올림)     | Baek_Seunghan_v03_ridge_logtarget_poly1_alpha100.csv | Log→Exp 역변환에서 에러가 지수적 증폭. 폭망. |
| 04  | 2026-05-30 | ElasticNet (α=10, l1_ratio=0.5) + StandardScaler | +0.02137 ± 0.01327 | **+0.0320**  | Baek_Seunghan_v04_elasticnet_a10_l1_05.csv | ⭐ 첫 양수. Ridge(α=10k), k-NN(k=200)도 같은 +0.021 수렴 → 신뢰가능. Gap=-0.003 (healthy). |
| 05  | 2026-05-30 | Lec07 변환(sqrt,제곱)+Ridge α=10000 | +0.01384 ± 0.02810 | (안올림)     | Baek_Seunghan_v05_lec07xform+ridge_a10000.csv | Feature engineering 시도. v4 못 넘음. |
| 06c | 2026-05-30 | k-NN(k=200, distance, raw features) | +0.02105 ± 0.02261 | **+0.02258** | Baek_Seunghan_v06c_ens_weighted_en00_ridge00_knn10.csv | k-NN 단독. ElasticNet보다 LB 낮음. calibration gap +0.005 (작음). |
| 07  | 2026-05-30 | Winsor[1-99] target + Ridge α=10 | +0.01502 ± 0.04148 | (오늘 올릴 예정) | Baek_Seunghan_v07_winsor1to99+ridge_a10.csv | 안전 winsor. 외곽 1% target만 clip. |
| **07b** | 2026-05-30 | **Winsor[0.5-99.5] target + Ridge α=1** | -0.02832 ± 0.16291 | **+0.04843** | Baek_Seunghan_v07b_winsor05to995+ridge_a1.csv | 🥈 **PUBLIC LB 1등 했었음 → 현재 2등.** OOF +0.0327 → LB +0.048 (calibration +0.016). Sweet spot 확정. |
| 08a | 2026-05-30 | Winsor[0.3-99.7] target + Ridge α=0.3 | -0.06091 ± 0.24180 | **+0.04116** | Baek_Seunghan_v08a_safe_winsor03to997_ridge_a03.csv | ⚠️ Sweet spot 살짝 바깥. OOF +0.0357 → LB +0.041 (gap +0.0055로 축소). Sweet spot 진짜 [0.5-99.5]에 단단히 박혀있음 증명. |
| 08b | 2026-05-30 | Winsor[0.2-99.8] target + Ridge α=0.3 | -0.09163 ± 0.30799 | **+0.03178** | Baek_Seunghan_v08b_risky_winsor02to998_ridge_a03.csv | ⚠️ v7d 데자뷰 증명. OOF +0.0381 → LB +0.032 (gap -0.006 음수). U-curve 완성. |
| 09  | 2026-05-30 | HGBR (HistGradientBoosting) sweep | best OOF +0.02736 (leaf=15 lr=0.05 sweet) | (미제출) | - | 계산만. Tree가 이 데이터에서 v7b 못 이김. **강의 외 모델 사용 결정** (과제 문서에 LightGBM 표준 명시). |
| 10c | 2026-05-30 | Multi-seed Ridge (s=0,42,100,2026)+Winsor[0.5-99.5] | OOF +0.0351 | **+0.04843** | Baek_Seunghan_v10c_multiseed_ridge_only.csv | LB가 v7b와 정확히 동일. Multi-seed는 LB 향상 0. |
| 10b | 2026-05-30 | Ensemble: Multi-seed Ridge×0.70 + HGBR-sweet×0.30 | OOF +0.0370 | **+0.03562** | Baek_Seunghan_v10b_ens_multiridge0.70_hgbr0.30.csv | ❌ HGBR blend가 LB -0.0128 폭락. Tree calibration gap 양수 X. Tree 완전 손 뗼. |
| **11a** | 2026-05-30 | **FE nonlinear (signed sqrt+log+square 15ft) + Winsor[0.5-99.5] + Ridge α=1** | **OOF +0.0410** | **+0.04190** | Baek_Seunghan_v11a_fe_nl_ridge_a1p0.csv | ❌ 괴멸적. OOF +0.0083 올랐는데 LB는 -0.0065 하락. **gap +0.016 → +0.001로 깊악**. 교훈: 이 데이터에서 OOF 향상은 거의 noise 포획. |
| 12 | 2026-06-01 | Phase 12: FE sqrt+log only + Ridge α0.3 + Winsor[0.5-99.5] | OOF +0.0437 | **+0.04702** | Baek_Seunghan_v12_abl_sqrt_log_a0p3.csv | square 제거해도 여전히 gap 대폭 깊악. # features가 단일 원인. |
| 13 | 2026-06-01 | QuantileTransformer(n=1000,normal) + Ridge α0.5 | OOF +0.0345 | **+0.04820** | Baek_Seunghan_v13_qnorm1000_a0p5.csv | 15 features 유지 → LB 거의 v7b와 동일. QuantileTransformer 효과 없음. |
| 14b | 2026-06-01 | v7b prediction × 2.0 (post-hoc rescaling) | OOF +0.0411 | **-0.03514** | Baek_Seunghan_v14b_rescale_x2p00.csv | ❌❌ **대재앙**. 극단값 방향 틀린 prediction을 4배 증폭. compression이 안전망이었음. |
| 14c | 2026-06-01 | Isotonic calibration on v7b OOF (CV-honest) | OOF -0.55 | (미제출) | - | CV-honest isotonic도 heavy-tail에서 폭망. Tail mapping 학습 불가. |
| 14d | 2026-06-01 | Compression sweep (x0.5~x1.10), winsor sweep, target QuantileTransform | OOF: x1.10 가 best (+0.0346) | (미제출) | - | 더 한 압축은 OOF 떨어뜨림. target QT도 괴멸. |
| 15 | 2026-06-01 | KernelRidge RBF (γ=0.001, α=0.01) + Winsor[0.5-99.5] | OOF +0.0344 | **+0.02256** | Baek_Seunghan_v15_krr_rbf_g0p001_a0p01.csv | ❌ kernel method도 속은다. mild nonlinearity조차 LB drop 큰. |
| 16 | 2026-06-01 | MEAN ensemble of LB-validated models (v7b + v10c + v13) | OOF +0.0342 | **+0.04838** | Baek_Seunghan_v16_mean_v7b_v10c_v13.csv | ✅ v7b와 사실상 동일 (-0.00005). Safe ensemble 입증. correlation 0.987+ 때문에 diversity 없어 이득 없음. |

| 17 | 2026-06-01 | **Phase 17 Day 1 exploration** (미제출) | - | - | v17_B_winsor_0.3_99.7 / v17_B_winsor_0.4_99.6 / v17_A_alpha_0.5 등 5개 saved | 5가지 방향 시도: α micro-sweep(무의미), Winsor band micro(OOF↑ 함정—v08a와 동일 패턴), PCA(정보 손실), Sample-weighted Ridge(다 worse), Drop extreme rows(다 worse). 진짜 신호 0개 발견. |

## 🎯 Phase 1–Phase 17 Meta-Lessons

1. **v7b family의 진짜 ceiling = 0.0484** (다수 시도로 확정)
2. **OOF 향상 ≠ LB 향상** 이 데이터에서 철칙으로 깨짐
3. **안전한 접근**: plain Ridge ± 동일 압축(winsor)만 안전
4. **폭망 자명**: FE 추가, tree, kernel(공격적 값), rescaling, isotonic, target QT, sample-weighted, PCA, row-drop
5. **Quantile Reg (median)** 은 heavy-tailed에서 R² 괴멸 → OOF +0.003
6. **Adversarial AUC = 0.489** 로 train/test 분포 같음 확인 (overfitting이 원인)
7. **Multi-seed/Mean ensemble**: high-correlation base들(>0.987) → 이득 없음, downside도 없음
8. **α 미세튜닝 무의미**: StandardScaler가 α를 normalize해서 0.5~2.0 동일
9. **Phase 17 Day 1 결론**: 5가지 새 방향 시도 → 진짜 신호 0개
10. **남은 시접: 진짜 다른 framing해야 함** (data 구조 diagnosis, two-stage, GP, stacking with meta-feature)

## 📦 남은 Submissions (3 days)

- Day 1–3: 총 15번
- 추천 사용: 8–11번 (과제솼용 아닌 과제답주뻗)
- 보존: 4–7번 (public LB overfit 방지)

## 🏆 Final 2 추천 (마감 전 이겁 설정)

- **Final 1**: v7b (가장 robust, baseline 0.04843 보장)
- **Final 2**: Day 1–3 중 LB 가장 높은 거 (도전적)



## 🏆 현재 순위 (Public LB)

| 순위 | 사람 | 점수 |
|---|---|---|
| 🥇 1 | Pawel Obrycki | 0.04877 |
| 🥈 2 | **Baek_Seunghan (v7b)** | **0.04843** |
| 🚩 | Baseline | 0.04599 |
| 3 | Tejasuk06 | 0.04623 |

⚠️ Public LB = test data의 51%만 사용. Private LB (나머지 49%)가 진짜 최종.

## 📊 Calibration gap 패턴 + Sweet spot U-curve 입증

| Winsor band | OOF R² | LB R² | Gap | 진단 |
|---|---|---|---|---|
| [0.1-99.9] (v7d, α=0.01) | +0.039 | +0.0190 | **-0.020** | 폭망 (variance) |
| [0.2-99.8] (v08b, α=0.3) | +0.0381 | (미제출) | 추정 < 0.030 | 던지지 않음 |
| [0.3-99.7] (v08a, α=0.3) | +0.0357 | +0.0412 | **+0.0055** | Gap 1/3로 축소 |
| **[0.5-99.5] (v7b, α=1)** | **+0.0327** | **+0.0484** | **+0.0157** | ⭐ Sweet spot |
| [1-99] (v7, α=10) | +0.023 | +0.0434 | +0.0204 | 안전 |

→ **OOF 증가 ≠ LB 증가.** Sweet spot [0.5-99.5]에서 정확히 calibration gap 최대화.
→ 발표 핵심 인사이트: **"OOF만 보고 모델 고르면 안 되는 이유"** 케이스 스터디.

## 📊 Calibration gap 패턴 (model family별)

| 모델 family | OOF R² → LB | Gap |
|---|---|---|
| 선형 (ElasticNet) v4 | +0.0140 → +0.0320 | **+0.018** |
| 선형 (Ridge+Winsor) v7b | +0.0327 → +0.04843 | **+0.0157** |
| k-NN v6c | +0.0177 → +0.02258 | **+0.005** |

→ 선형 모델 = Calibration gap 크게 (+0.015~0.018)
→ k-NN = Calibration gap 작게 (+0.005)

## 📊 Phase 18-19 (2026-06-02) — Genuinely Different Framings Test

### Phase 18: A-E test (모두 0개 신호)
| 후보 | 최고 OOF | corr(v7b) | 결과 |
|---|---|---|---|
| A. Huber | +0.0036 | 0.42 | 💀 Robust loss < winsor |
| B. X-winsor | +0.0274 | 0.997 | 🟡 v7b 복사본 |
| C. Residual stack | +0.0213 | 0.94 | 💀 Residual std=1945 = 순수 noise |
| D. TheilSen | +0.0313 | 0.9996 | 🟡 본질적으로 Ridge |
| E. Rank-based | -31 | 0.25 | 💀💀 inverse-transform 폭망 |

**핵심 발견**: Residual std 1945 vs mean 5.7 → linear signal 다 뽑힘 입증.

### Phase 19: F-I test (1개 의심스러운 후보)
| 후보 | 최고 OOF | corr(v7b) | 결과 |
|---|---|---|---|
| F. YeoJohnson | +0.0326 | 0.992 | 🟡 v7b 그 자체 |
| G. PLS | +0.0327 | 1.0000 | 🟡 수학적으로 Ridge와 동일 |
| H. Per-quantile | -0.074 | 0.79 | 💀 fold당 sample 부족 |
| I. magtail × 1.5 | +0.0422 | 0.985 | ⚠️ v14b 데자뷰 (위험!) |
| I. magtail × 1.2 | +0.0370 | 0.997 | 실험으로 던짐 |


## 📊 Phase 20 (2026-06-02) — Symmetric Perturbation Test

| 실험 | OOF | LB | v7b 대비 |
|---|---|---|---|
| v7b baseline | +0.0327 | **0.04843** | — |
| v19 magtail ×1.2 (tail expand) | +0.0370 | 0.04731 | -0.00112 |
| v20 magtrim ×0.9 (tail shrink) | +0.0303 | 0.04720 | **-0.00123** |

### 🔑 결정적 발견
**Tail을 양방향 어디로 건드려도 LB는 거의 동일하게 -0.001 손해**

→ v7b는 **진짜 local optimum**. tails가 이미 optimally calibrated.
→ 어떤 post-hoc tail adjustment도 LB 향상 불가.
→ 20 phases 끝에 v7b = 0.04843이 우리 toolkit의 ceiling 확정.

## 🎯 최종 발표 메시지

"OOF를 보고 모델을 선택하지 마세요"의 케이스 스터디:
- Phase 8: Sweet spot U-curve (winsor band)
- Phase 11: FE OOF↑ → LB↓
- Phase 14b: Rescaling OOF↑ → LB 폭망
- Phase 19-20: Symmetric magtail/magtrim 양방향 손해

→ **Calibration gap을 인식하지 못하면 OOF는 거짓말이다**

