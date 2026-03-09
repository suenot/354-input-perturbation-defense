use anyhow::Result;
use input_perturbation_defense::*;
use ndarray::Array1;

fn main() -> Result<()> {
    println!("=== Chapter 224: Input Perturbation Defense for Trading ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch BTCUSDT data from Bybit
    // -----------------------------------------------------------------------
    println!("[1] Fetching BTCUSDT 1-minute klines from Bybit...");
    let close_prices = match fetch_bybit_klines("BTCUSDT", "1", 200) {
        Ok(prices) => {
            println!("    Fetched {} data points", prices.len());
            println!(
                "    Price range: {:.2} - {:.2}",
                prices.iter().cloned().fold(f64::INFINITY, f64::min),
                prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            );
            prices
        }
        Err(e) => {
            println!("    Warning: Could not fetch from Bybit ({})", e);
            println!("    Using synthetic data instead...");
            generate_synthetic_prices(200)
        }
    };

    // Normalize to [0, 1]
    let (normalized, price_min, price_max) = normalize(&close_prices);
    println!(
        "    Normalized {} prices to [0, 1]\n",
        normalized.len()
    );

    // -----------------------------------------------------------------------
    // Step 2: Create sample windows for evaluation
    // -----------------------------------------------------------------------
    println!("[2] Creating evaluation dataset...");
    let window_size = 20;
    let threshold = 0.5;
    let mut windows: Vec<Array1<f64>> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();

    for i in 0..normalized.len().saturating_sub(window_size) {
        let window = normalized.slice(ndarray::s![i..i + window_size]).to_owned();
        let label = if window.mean().unwrap_or(0.0) >= threshold {
            1
        } else {
            0
        };
        windows.push(window);
        labels.push(label);
    }
    println!(
        "    Created {} windows of size {}",
        windows.len(),
        window_size
    );
    let n_positive: usize = labels.iter().map(|&l| l as usize).sum();
    println!(
        "    Labels: {} positive, {} negative\n",
        n_positive,
        labels.len() - n_positive
    );

    // -----------------------------------------------------------------------
    // Step 3: Generate adversarial examples
    // -----------------------------------------------------------------------
    println!("[3] Generating adversarial examples...");
    let epsilon = 0.05;
    let pgd_alpha = 0.01;
    let pgd_steps = 10;

    let fgsm_adversarials: Vec<Array1<f64>> = windows
        .iter()
        .zip(labels.iter())
        .map(|(w, &l)| {
            let direction = if l == 1 { -1.0 } else { 1.0 };
            fgsm_attack(w, epsilon, direction)
        })
        .collect();

    let pgd_adversarials: Vec<Array1<f64>> = windows
        .iter()
        .zip(labels.iter())
        .map(|(w, &l)| {
            let direction = if l == 1 { -1.0 } else { 1.0 };
            pgd_attack(w, epsilon, pgd_alpha, pgd_steps, direction)
        })
        .collect();

    // Measure attack success rate
    let fgsm_flipped: usize = windows
        .iter()
        .zip(fgsm_adversarials.iter())
        .zip(labels.iter())
        .filter(|((orig, adv), &label)| {
            threshold_classify(orig, threshold) == label
                && threshold_classify(adv, threshold) != label
        })
        .count();

    let pgd_flipped: usize = windows
        .iter()
        .zip(pgd_adversarials.iter())
        .zip(labels.iter())
        .filter(|((orig, adv), &label)| {
            threshold_classify(orig, threshold) == label
                && threshold_classify(adv, threshold) != label
        })
        .count();

    println!("    Epsilon: {}", epsilon);
    println!(
        "    FGSM attack success: {}/{} ({:.1}%)",
        fgsm_flipped,
        windows.len(),
        100.0 * fgsm_flipped as f64 / windows.len() as f64
    );
    println!(
        "    PGD attack success:  {}/{} ({:.1}%)\n",
        pgd_flipped,
        windows.len(),
        100.0 * pgd_flipped as f64 / windows.len() as f64
    );

    // -----------------------------------------------------------------------
    // Step 4: Evaluate each defense method
    // -----------------------------------------------------------------------
    println!("[4] Evaluating defense methods against FGSM attack...");
    println!(
        "    {:<22} | {:<12} | {:<24} | {:<20}",
        "Defense", "Clean Acc", "Robust (no defense)", "Robust (defended)"
    );
    println!("    {}", "-".repeat(85));

    // No defense baseline
    let no_defense_clean: f64 = windows
        .iter()
        .zip(labels.iter())
        .filter(|(w, &l)| threshold_classify(w, threshold) == l)
        .count() as f64
        / windows.len() as f64;

    let no_defense_robust: f64 = fgsm_adversarials
        .iter()
        .zip(labels.iter())
        .filter(|(adv, &l)| threshold_classify(adv, threshold) == l)
        .count() as f64
        / windows.len() as f64;

    println!(
        "    {:<22} | {:<12.1}% | {:<24.1}% | {:<20}",
        "No Defense",
        no_defense_clean * 100.0,
        no_defense_robust * 100.0,
        "N/A"
    );

    // Define defenses to test
    let defenses: Vec<Box<dyn Defense>> = vec![
        Box::new(MovingAverageDefense::new(2)),
        Box::new(MovingAverageDefense::new(5)),
        Box::new(GaussianSmoothingDefense::new(1.0)),
        Box::new(GaussianSmoothingDefense::new(2.0)),
        Box::new(BitDepthReductionDefense::new(0.1)),
        Box::new(BitDepthReductionDefense::new(0.05)),
        Box::new(FeatureSqueezingDefense::new(0.1, 2)),
    ];

    for defense in &defenses {
        let eval = evaluate_defense(defense.as_ref(), &windows, &labels, epsilon, threshold);
        println!(
            "    {:<22} | {:<12.1}% | {:<24.1}% | {:<20.1}%",
            eval.defense_name,
            eval.clean_accuracy * 100.0,
            eval.robust_accuracy * 100.0,
            eval.clean_defended_accuracy * 100.0,
        );
    }

    // -----------------------------------------------------------------------
    // Step 5: Randomized Smoothing with certified radius
    // -----------------------------------------------------------------------
    println!("\n[5] Randomized Smoothing with certified radius...");
    let rs_defense = RandomizedSmoothingDefense::new(0.05, 200, threshold);

    let n_demo = 5.min(windows.len());
    println!("    Evaluating {} sample windows:\n", n_demo);
    println!(
        "    {:<8} | {:<10} | {:<10} | {:<12} | {:<10} | {:<10}",
        "Window", "Clean", "Adv", "Smoothed", "Confidence", "Cert.Radius"
    );
    println!("    {}", "-".repeat(70));

    for i in 0..n_demo {
        let clean_pred = threshold_classify(&windows[i], threshold);
        let adv_pred = threshold_classify(&fgsm_adversarials[i], threshold);
        let (smoothed_pred, confidence) = rs_defense.smoothed_classify(&windows[i]);
        let radius = rs_defense.certified_radius(confidence);

        println!(
            "    {:<8} | {:<10} | {:<10} | {:<12} | {:<10.3} | {:<10.4}",
            i, clean_pred, adv_pred, smoothed_pred, confidence, radius
        );
    }

    // -----------------------------------------------------------------------
    // Step 6: Thermometer Encoding demonstration
    // -----------------------------------------------------------------------
    println!("\n[6] Thermometer Encoding demonstration...");
    let thermo = ThermometerEncoding::new(10);

    if let Some(sample) = windows.first() {
        let first_3: Array1<f64> = sample.slice(ndarray::s![0..3]).to_owned();
        println!("    Original values: {:?}", first_3.to_vec());

        let encoded = thermo.encode(&first_3);
        println!("    Encoded length: {} (3 values x 10 levels)", encoded.len());

        let decoded = thermo.decode(&encoded);
        println!("    Decoded values:  {:?}", decoded.to_vec());

        // Show robustness: perturb and encode
        let perturbed = fgsm_attack(&first_3, 0.03, 1.0);
        println!("\n    Perturbed values: {:?}", perturbed.to_vec());

        let encoded_pert = thermo.encode(&perturbed);
        let decoded_pert = thermo.decode(&encoded_pert);
        println!("    Decoded perturbed: {:?}", decoded_pert.to_vec());

        let diff: f64 = decoded
            .iter()
            .zip(decoded_pert.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        println!("    Total absolute difference after encoding: {:.4}", diff);
    }

    // -----------------------------------------------------------------------
    // Step 7: Price-space example
    // -----------------------------------------------------------------------
    println!("\n[7] Defense in price space (denormalized)...");
    if let Some(sample) = windows.first() {
        let prices = denormalize(sample, price_min, price_max);
        let adv_norm = fgsm_attack(sample, epsilon, -1.0);
        let adv_prices = denormalize(&adv_norm, price_min, price_max);

        let ma_defense = MovingAverageDefense::new(2);
        let defended_norm = ma_defense.apply(&adv_norm);
        let defended_prices = denormalize(&defended_norm, price_min, price_max);

        println!("    First 5 values comparison:");
        println!(
            "    {:<12} | {:<14} | {:<14} | {:<14}",
            "Index", "Original ($)", "Attacked ($)", "Defended ($)"
        );
        println!("    {}", "-".repeat(60));
        for j in 0..5.min(prices.len()) {
            println!(
                "    {:<12} | {:<14.2} | {:<14.2} | {:<14.2}",
                j, prices[j], adv_prices[j], defended_prices[j]
            );
        }
    }

    println!("\n=== Done! ===");
    Ok(())
}

/// Generate synthetic price data when API is unavailable.
fn generate_synthetic_prices(n: usize) -> Vec<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(n);
    let mut price: f64 = 50000.0;
    for _ in 0..n {
        price += rng.gen_range(-100.0..100.0);
        price = price.max(40000.0).min(60000.0);
        prices.push(price);
    }
    prices
}
