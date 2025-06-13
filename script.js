$(document).ready(function() {
    let drugFoodData = [];

    // Load CSV data
    async function loadCSVData() {
        try {
            const csvContent = await window.fs.readFile('/Users/sachidhoka/Desktop/balanced_drug_food_interactions.csv', { encoding: 'utf8' });
            const parsedData = Papa.parse(csvContent, {
                header: true,
                skipEmptyLines: true,
                dynamicTyping: true
            });
            drugFoodData = parsedData.data;
            console.log('CSV data loaded:', drugFoodData.length, 'records');
        } catch (error) {
            console.error('Error loading CSV:', error);
        }
    }

    // Call the function to load data
    loadCSVData();

    // Initialize Select2 for drug search
    $('#drug').select2({
        placeholder: "Search for a medication...",
        allowClear: true,
        data: function() {
            const uniqueDrugs = [...new Set(drugFoodData.map(row => row.drug))];
            return uniqueDrugs.map(drug => ({
                id: drug,
                text: drug,
                category: drugFoodData.find(row => row.drug === drug)?.drug_category || ''
            }));
        }(),
        templateResult: formatDrugResult,
        templateSelection: formatDrugSelection
    });

    // Initialize Select2 for food search
    $('#food').select2({
        placeholder: "Search for a food or supplement...",
        allowClear: true,
        data: function() {
            const uniqueFoods = [...new Set(drugFoodData.map(row => row.food))];
            return uniqueFoods.map(food => ({
                id: food,
                text: food,
                category: drugFoodData.find(row => row.food === food)?.food_category || ''
            }));
        }(),
        templateResult: formatFoodResult,
        templateSelection: formatFoodSelection
    });

    // Format drug search results
    function formatDrugResult(drug) {
        if (!drug.id) {
            return drug.text;
        }
        
        var $result = $(
            '<div class="select2-result">' +
                '<div class="select2-result__title">' + drug.text + '</div>' +
                (drug.category ? '<div class="select2-result__category">' + drug.category + '</div>' : '') +
            '</div>'
        );
        
        return $result;
    }

    // Format drug selection display
    function formatDrugSelection(drug) {
        if (!drug.id) {
            return drug.text;
        }
        
        return $(
            '<div class="selected-item">' +
                '<span class="selected-item__name">' + drug.text + '</span>' +
                (drug.category ? '<span class="selected-item__category">' + drug.category + '</span>' : '') +
            '</div>'
        );
    }

    // Format food search results
    function formatFoodResult(food) {
        if (!food.id) {
            return food.text;
        }
        
        var $result = $(
            '<div class="select2-result">' +
                '<div class="select2-result__title">' + food.text + '</div>' +
                (food.category ? '<div class="select2-result__category">' + food.category + '</div>' : '') +
            '</div>'
        );
        
        return $result;
    }

    // Format food selection display
    function formatFoodSelection(food) {
        if (!food.id) {
            return food.text;
        }
        
        return $(
            '<div class="selected-item">' +
                '<span class="selected-item__name">' + food.text + '</span>' +
                (food.category ? '<span class="selected-item__category">' + food.category + '</span>' : '') +
            '</div>'
        );
    }

    // Clear form button
    $('.btn-clear').click(function() {
        $('#drug').val(null).trigger('change');
        $('#food').val(null).trigger('change');
        $('#output').html(`
            <div class="results-placeholder">
                <i class="fas fa-microscope"></i>
                <p>Enter a drug and food combination to see interaction analysis</p>
            </div>
        `);
    });

    // Handle form submission
    $('#interactionForm').on('submit', function(e) {
        e.preventDefault();
        
        const drugId = $('#drug').val();
        const drugText = $('#drug').select2('data')[0]?.text || '';
        const foodId = $('#food').val();
        const foodText = $('#food').select2('data')[0]?.text || '';
        
        const outputDiv = $('#output');
        
        if (!drugId || !foodId) {
            outputDiv.html('<div class="alert alert-danger">Please select both a medication and a food item.</div>');
            return;
        }
        
        // Show loading state
        outputDiv.html(`
            <div class="loading-state">
                <div class="spinner"></div>
                <p>Analyzing interaction between <strong>${drugText}</strong> and <strong>${foodText}</strong>...</p>
            </div>
        `);
        
        // Find actual interaction data instead of mock response
        const actualInteraction = drugFoodData.find(row => 
            row.drug === drugText && row.food === foodText
        );

        if (actualInteraction) {
            displayPredictionResults({
                drug: actualInteraction.drug,
                food: actualInteraction.food,
                interaction_predicted: actualInteraction.interaction === 1 || actualInteraction.interaction === true,
                probability: actualInteraction.interaction === 1 ? 0.95 : 0.05,
                drug_category: actualInteraction.drug_category,
                food_category: actualInteraction.food_category,
                mechanism: actualInteraction.mechanism,
                risk_level: actualInteraction.risk_level,
                confidence: 'High',
                explanation: getMechanismExplanation(actualInteraction.mechanism, actualInteraction.drug_category, actualInteraction.food_category),
                recommendations: getRecommendation(actualInteraction.mechanism, actualInteraction.risk_level)
            });
        } else {
            displayPredictionResults({
                drug: drugText,
                food: foodText,
                interaction_predicted: false,
                probability: 0.05,
                drug_category: 'unknown',
                food_category: 'unknown',
                mechanism: 'none',
                risk_level: 'LOW',
                confidence: 'Low',
                explanation: 'No interaction data found for this combination.',
                recommendations: 'Consult healthcare provider for guidance.'
            });
        }
    });
    
    function getMechanismExplanation(mechanism, drugCat, foodCat) {
        const explanations = {
            'cyp3a4_inhibition': `Components in ${foodCat} inhibit CYP3A4 enzymes in the liver, which are responsible for metabolizing ${drugCat} medications. This can lead to increased drug levels in the blood.`,
            'vitamin_k_competition': `${foodCat} contains vitamin K which opposes the blood-thinning effects of ${drugCat} medications by supporting clotting factor production.`,
            'calcium_chelation': `Calcium in ${foodCat} binds to ${drugCat}, forming insoluble complexes that reduce drug absorption in the gastrointestinal tract.`,
            'absorption_interference': `${foodCat} may delay or reduce the absorption of ${drugCat} by affecting gastric emptying or forming physical barriers in the GI tract.`,
            'none': 'No known pharmacokinetic or pharmacodynamic interaction between these substances.'
        };
        
        return explanations[mechanism] || 'Potential interaction through an unspecified mechanism.';
    }
    
    function getRecommendation(mechanism, risk) {
        const recommendations = {
            'HIGH': {
                'cyp3a4_inhibition': 'Avoid consuming these together. Consider alternative medications or foods.',
                'vitamin_k_competition': 'Maintain consistent vitamin K intake. Monitor INR closely and adjust warfarin dose as needed.',
                'default': 'Avoid combination. Consult healthcare provider for alternatives.'
            },
            'MODERATE': {
                'calcium_chelation': 'Take medication 2 hours before or 4 hours after consuming calcium-rich foods.',
                'absorption_interference': 'Take medication on an empty stomach if possible, or at consistent times relative to meals.',
                'default': 'Space administration times. Monitor for reduced efficacy or side effects.'
            },
            'LOW': {
                'default': 'Minimal clinical significance. No special precautions required for most patients.'
            }
        };
        
        return recommendations[risk][mechanism] || recommendations[risk]['default'] || 'Monitor for any unexpected effects.';
    }
    
    // Display prediction results
    function displayPredictionResults(data) {
        const outputDiv = $('#output');
        
        if (data.error) {
            outputDiv.html(`
                <div class="error-state">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h3>Error Analyzing Interaction</h3>
                    <p>${data.error}</p>
                    <button class="btn-try-again">Try Again</button>
                </div>
            `);
            return;
        }
        
        const riskClass = data.risk_level.toLowerCase();
        const mechanismText = data.mechanism.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        
        let mechanismDetails = '';
        if (data.mechanism !== 'none') {
            mechanismDetails = `
                <div class="mechanism-details">
                    <h4><i class="fas fa-microscope"></i> Interaction Mechanism</h4>
                    <p>${data.explanation}</p>
                    <div class="mechanism-tag">${mechanismText}</div>
                </div>
            `;
        }
        
        let similarInteractions = '';
        if (data.interaction_predicted) {
            similarInteractions = `
                <div class="similar-interactions">
                    <h4><i class="fas fa-random"></i> Similar Interactions</h4>
                    <div class="similar-list">
                        <div class="similar-item">
                            <div class="similar-pair">Warfarin + Kale</div>
                            <div class="similar-risk high-risk">High Risk</div>
                        </div>
                        <div class="similar-item">
                            <div class="similar-pair">Simvastatin + Pomelo</div>
                            <div class="similar-risk high-risk">High Risk</div>
                        </div>
                        <div class="similar-item">
                            <div class="similar-pair">Tetracycline + Yogurt</div>
                            <div class="similar-risk moderate-risk">Moderate Risk</div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        outputDiv.html(`
            <div class="prediction-header">
                <div class="drug-food-pair">
                    <span class="drug-name">${data.drug}</span>
                    <span class="interaction-plus">+</span>
                    <span class="food-name">${data.food}</span>
                </div>
                <div class="prediction-confidence">
                    <span class="confidence-tag">${data.confidence} Confidence</span>
                </div>
            </div>
            
            <div class="prediction-summary ${riskClass}-risk">
                <div class="risk-level">
                    <div class="risk-icon">
                        ${data.risk_level === 'HIGH' ? '<i class="fas fa-exclamation-triangle"></i>' : 
                          data.risk_level === 'MODERATE' ? '<i class="fas fa-info-circle"></i>' : 
                          '<i class="fas fa-check-circle"></i>'}
                    </div>
                    <div class="risk-text">
                        <h3>${data.interaction_predicted ? 'Interaction Detected' : 'No Significant Interaction'}</h3>
                        <p class="risk-category ${riskClass}-risk">${data.risk_level} Risk</p>
                    </div>
                    <div class="risk-probability">
                        <div class="probability-circle">
                            <svg viewBox="0 0 36 36" class="circular-chart">
                                <path class="circle-bg"
                                    d="M18 2.0845
                                    a 15.9155 15.9155 0 0 1 0 31.831
                                    a 15.9155 15.9155 0 0 1 0 -31.831"
                                />
                                <path class="circle-fill"
                                    stroke-dasharray="${Math.round(data.probability * 100)}, 100"
                                    d="M18 2.0845
                                    a 15.9155 15.9155 0 0 1 0 31.831
                                    a 15.9155 15.9155 0 0 1 0 -31.831"
                                />
                                <text x="18" y="20.5" class="percentage">${Math.round(data.probability * 100)}%</text>
                            </svg>
                        </div>
                    </div>
                </div>
                
                <div class="category-tags">
                    <span class="drug-category">${data.drug_category.replace(/_/g, ' ').toUpperCase()}</span>
                    <span class="food-category">${data.food_category.replace(/_/g, ' ').toUpperCase()}</span>
                </div>
                
                ${mechanismDetails}
                
                <div class="clinical-recommendations">
                    <h4><i class="fas fa-stethoscope"></i> Clinical Recommendations</h4>
                    <p>${data.recommendations}</p>
                </div>
                
                ${similarInteractions}
                
                <div class="prediction-actions">
                    <button class="btn-save-to-profile">
                        <i class="fas fa-save"></i> Save to Profile
                    </button>
                    <button class="btn-generate-report">
                        <i class="fas fa-file-pdf"></i> Generate PDF Report
                    </button>
                    <button class="btn-explain-more">
                        <i class="fas fa-question-circle"></i> Explain More
                    </button>
                </div>
            </div>
        `);
        
        // Add event listeners for action buttons
        $('.btn-save-to-profile').click(function() {
            alert('Please login to save this interaction to your profile.');
        });
        
        $('.btn-generate-report').click(function() {
            alert('PDF report generation would be implemented here.');
        });
        
        $('.btn-explain-more').click(function() {
            alert('Detailed explanation would be shown here.');
        });
    }
    
    // Login/Signup Modal Toggle
    $('.btn-login').click(function() {
        $('#loginModal').addClass('active');
    });
    
    $('.btn-signup').click(function() {
        $('#signupModal').addClass('active');
    });
    
    $('.modal-close, .switch-to-signup').click(function() {
        $('#loginModal').removeClass('active');
        $('#signupModal').addClass('active');
    });
    
    $('.switch-to-login').click(function() {
        $('#signupModal').removeClass('active');
        $('#loginModal').addClass('active');
    });
    
    // Close modal when clicking outside
    $(document).click(function(e) {
        if ($(e.target).hasClass('modal')) {
            $('.modal').removeClass('active');
        }
    });
    
    // Mobile menu toggle
    $('.mobile-menu-btn').click(function() {
        $('.nav-links').toggleClass('active');
    });
    
    // Smooth scrolling for navigation links
    $('a[href^="#"]').on('click', function(e) {
        e.preventDefault();
        
        $('.nav-links').removeClass('active');
        
        const target = $(this).attr('href');
        if (target === '#') return;
        
        $('html, body').animate({
            scrollTop: $(target).offset().top - 80
        }, 500);
    });
    
    // Add active class to nav links based on scroll position
    $(window).scroll(function() {
        const scrollPosition = $(this).scrollTop();
        
        $('section').each(function() {
            const currentSection = $(this);
            const sectionTop = currentSection.offset().top - 100;
            const sectionId = currentSection.attr('id');
            
            if (scrollPosition >= sectionTop && 
                scrollPosition < sectionTop + currentSection.outerHeight()) {
                $('.nav-links a').removeClass('active');
                $('.nav-links a[href="#' + sectionId + '"]').addClass('active');
            }
        });
    });
    
    // Initialize with first section active
    $('.nav-links a[href="#predictor"]').addClass('active');
});
