$(document).ready(function() {
    // Initialize Select2 for drug search
    $('#drug').select2({
        placeholder: "Search for a medication...",
        allowClear: true,
        ajax: {
            url: '/search/drugs',
            dataType: 'json',
            delay: 250,
            data: function(params) {
                return {
                    q: params.term
                };
            },
            processResults: function(data) {
                return {
                    results: data
                };
            },
            cache: true,
            minimumInputLength: 2
        },
        templateResult: formatDrugResult,
        templateSelection: formatDrugSelection
    });

    // Initialize Select2 for food search
    $('#food').select2({
        placeholder: "Search for a food or supplement...",
        allowClear: true,
        ajax: {
            url: '/search/foods',
            dataType: 'json',
            delay: 250,
            data: function(params) {
                return {
                    q: params.term
                };
            },
            processResults: function(data) {
                return {
                    results: data
                };
            },
            cache: true,
            minimumInputLength: 2
        },
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
        
        // Simulate API call (replace with actual AJAX call to your backend)
        setTimeout(() => {
            // This is a mock response - replace with actual API response
            const mockResponse = generateMockPrediction(drugText, foodText);
            displayPredictionResults(mockResponse);
        }, 1500);
    });
    
    // Generate mock prediction data (replace with actual API call)
    function generateMockPrediction(drug, food) {
        const riskLevels = ['HIGH', 'MODERATE', 'LOW'];
        const mechanisms = [
            'cyp3a4_inhibition',
            'vitamin_k_competition',
            'calcium_chelation',
            'absorption_interference',
            'none'
        ];
        
        const drugCategory = drug.toLowerCase().includes('warfarin') ? 'anticoagulant' : 
                            drug.toLowerCase().includes('statin') ? 'statin' : 
                            drug.toLowerCase().includes('antibiotic') ? 'antibiotic' : 
                            'other';
        
        const foodCategory = food.toLowerCase().includes('grapefruit') ? 'citrus' : 
                            food.toLowerCase().includes('spinach') ? 'leafy_greens' : 
                            food.toLowerCase().includes('milk') ? 'dairy' : 
                            'other';
        
        // Special cases for known interactions
        if ((drugCategory === 'anticoagulant' && foodCategory === 'leafy_greens') ||
            (drugCategory === 'statin' && foodCategory === 'citrus')) {
            return {
                drug: drug,
                food: food,
                interaction_predicted: true,
                probability: 0.95,
                drug_category: drugCategory,
                food_category: foodCategory,
                mechanism: drugCategory === 'anticoagulant' ? 'vitamin_k_competition' : 'cyp3a4_inhibition',
                risk_level: 'HIGH',
                confidence: 'High',
                explanation: drugCategory === 'anticoagulant' ? 
                    'Vitamin K in leafy greens can reduce the effectiveness of anticoagulants like warfarin.' : 
                    'Grapefruit inhibits CYP3A4 enzymes, increasing statin levels in the blood.',
                recommendations: drugCategory === 'anticoagulant' ? 
                    'Maintain consistent vitamin K intake and monitor INR regularly.' : 
                    'Avoid grapefruit products completely while taking this medication.'
            };
        }
        
        // Random generation for other cases
        const interaction = Math.random() > 0.7;
        const mechanism = interaction ? mechanisms[Math.floor(Math.random() * (mechanisms.length - 1))] : 'none';
        const risk = interaction ? riskLevels[Math.floor(Math.random() * riskLevels.length)] : 'LOW';
        
        return {
            drug: drug,
            food: food,
            interaction_predicted: interaction,
            probability: interaction ? (0.6 + Math.random() * 0.4) : Math.random() * 0.4,
            drug_category: drugCategory,
            food_category: foodCategory,
            mechanism: mechanism,
            risk_level: risk,
            confidence: interaction ? (Math.random() > 0.5 ? 'High' : 'Medium') : 'High',
            explanation: interaction ? 
                getMechanismExplanation(mechanism, drugCategory, foodCategory) : 
                'No significant interaction expected between these substances.',
            recommendations: interaction ? 
                getRecommendation(mechanism, risk) : 
                'No special precautions needed.'
        };
    }
    
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
    
    // Chatbot functionality (mock)
    $('.chatbot-input button').click(function() {
        const input = $('.chatbot-input input');
        const message = input.val().trim();
        
        if (message) {
            // Add user message
            $('.chatbot-messages').append(`
                <div class="message-user">
                    <div class="message-content">
                        ${message}
                    </div>
                </div>
            `);
            
            // Clear input
            input.val('');
            
            // Show typing indicator
            $('.chatbot-messages').append(`
                <div class="message-bot typing">
                    <div class="message-content">
                        <div class="typing-indicator">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>
                </div>
            `);
            
            // Scroll to bottom
            $('.chatbot-messages').scrollTop($('.chatbot-messages')[0].scrollHeight);
            
            // Simulate bot response after delay
            setTimeout(() => {
                $('.typing').remove();
                addBotResponse(message);
            }, 1500);
        }
    });
    
    // Also trigger on Enter key
    $('.chatbot-input input').keypress(function(e) {
        if (e.which === 13) {
            $('.chatbot-input button').click();
        }
    });
    
    // Add bot response based on user message
    function addBotResponse(message) {
        let response = '';
        const lowerMessage = message.toLowerCase();
        
        if (lowerMessage.includes('warfarin') || lowerMessage.includes('coumadin')) {
            response = `Warfarin (Coumadin) interacts significantly with vitamin K-rich foods like leafy greens (spinach, kale). These foods can reduce warfarin's effectiveness. It's important to maintain consistent vitamin K intake rather than avoid it completely.`;
        } 
        else if (lowerMessage.includes('statin') || lowerMessage.includes('atorvastatin') || lowerMessage.includes('simvastatin')) {
            response = `Statins like atorvastatin and simvastatin can interact with grapefruit and grapefruit juice. Grapefruit inhibits CYP3A4 enzymes in the liver, which can increase statin levels in the blood and raise the risk of side effects like muscle pain.`;
        }
        else if (lowerMessage.includes('antibiotic') || lowerMessage.includes('tetracycline') || lowerMessage.includes('doxycycline')) {
            response = `Certain antibiotics, particularly tetracyclines and fluoroquinolones, can interact with dairy products, calcium supplements, and antacids. Calcium binds to these antibiotics, reducing their absorption. Take these antibiotics 2 hours before or 4 hours after consuming dairy.`;
        }
        else if (lowerMessage.includes('coffee') || lowerMessage.includes('caffeine')) {
            response = `Caffeine can interact with several medications. It may increase side effects of stimulants, reduce absorption of thyroid medications, and interact with certain antidepressants. The effects vary by medication.`;
        }
        else if (lowerMessage.includes('alcohol')) {
            response = `Alcohol can interact dangerously with many medications including pain relievers, antidepressants, anxiety medications, and more. It can increase sedation, liver toxicity, or reduce medication effectiveness. Always check with your pharmacist about alcohol interactions.`;
        }
        else {
            response = `I can provide information about how specific foods interact with medications. Try asking about a particular drug (like "warfarin" or "statins") or food (like "grapefruit" or "dairy"). You can also use the interaction predictor tool above for detailed analysis.`;
        }
        
        $('.chatbot-messages').append(`
            <div class="message-bot">
                <div class="message-content">
                    ${response}
                    <div class="quick-replies">
                        <button class="quick-reply">More about warfarin</button>
                        <button class="quick-reply">Statin interactions</button>
                        <button class="quick-reply">Antibiotic timing</button>
                    </div>
                </div>
            </div>
        `);
        
        // Scroll to bottom
        $('.chatbot-messages').scrollTop($('.chatbot-messages')[0].scrollHeight);
        
        // Add click handler for quick replies
        $('.quick-reply').click(function() {
            $('.chatbot-input input').val($(this).text());
            $('.chatbot-input button').click();
        });
    }
});