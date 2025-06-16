

$(document).ready(function() {
    let API_BASE_URL = ''; // Use relative URLs for same domain
    
    // Check if API is running
    function checkAPIStatus() {
        $.ajax({
            url: '/api/status',
            type: 'GET',
            success: function(response) {
                console.log('API Status:', response);
                initializeSelect2();
            },
            error: function(xhr) {
                console.error('API not responding:', xhr);
                showAlert('Backend API is not responding. Please check if the server is running.', 'error');
            }
        });
    }

    // Initialize Select2 dropdowns with AJAX data loading
    function initializeSelect2() {
        // Drug dropdown with AJAX
        $('#drug').select2({
            placeholder: "Type to search medications...",
            allowClear: true,
            width: '100%',
            minimumInputLength: 1,
            ajax: {
                url: '/api/drugs',
                dataType: 'json',
                delay: 250,
                data: function (params) {
                    return {
                        q: params.term, // search term
                        limit: 20
                    };
                },
                processResults: function (data) {
                    return {
                        results: data.drugs.map(function(drug) {
                            return {
                                id: drug,
                                text: drug.charAt(0).toUpperCase() + drug.slice(1) // Capitalize first letter
                            };
                        })
                    };
                },
                cache: true
            }
        });

        // Food dropdown with AJAX
        $('#food').select2({
            placeholder: "Type to search foods...",
            allowClear: true,
            width: '100%',
            minimumInputLength: 1,
            ajax: {
                url: '/api/foods',
                dataType: 'json',
                delay: 250,
                data: function (params) {
                    return {
                        q: params.term, // search term
                        limit: 20
                    };
                },
                processResults: function (data) {
                    return {
                        results: data.foods.map(function(food) {
                            return {
                                id: food,
                                text: food.charAt(0).toUpperCase() + food.slice(1) // Capitalize first letter
                            };
                        })
                    };
                },
                cache: true
            }
        });

        // Load initial data for dropdowns (optional - show some options before typing)
        loadInitialData();
    }

    // Load initial data to show some options before user types
    function loadInitialData() {
        // Load initial drugs
        $.ajax({
            url: '/api/drugs',
            type: 'GET',
            data: { limit: 10 },
            success: function(response) {
                const drugOptions = response.drugs.map(function(drug) {
                    return new Option(drug.charAt(0).toUpperCase() + drug.slice(1), drug, false, false);
                });
                $('#drug').append(drugOptions).trigger('change');
            }
        });

        // Load initial foods
        $.ajax({
            url: '/api/foods',
            type: 'GET',
            data: { limit: 10 },
            success: function(response) {
                const foodOptions = response.foods.map(function(food) {
                    return new Option(food.charAt(0).toUpperCase() + food.slice(1), food, false, false);
                });
                $('#food').append(foodOptions).trigger('change');
            }
        });
    }

    // Start by checking API status
    checkAPIStatus();

    // Handle form submission
    $('#interactionForm').on('submit', function(e) {
        e.preventDefault();
        
        const drug = $('#drug').val();
        const food = $('#food').val();
        
        if (!drug || !food) {
            showAlert('Please select both a medication and a food item', 'error');
            return;
        }
        
        // Show loading state
        showLoading(drug, food);
        
        // Send prediction request to backend
        $.ajax({
            url: '/api/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                drug: drug,
                food: food
            }),
            success: function(response) {
                if (response.success) {
                    displayPredictionResults(response.data);
                } else {
                    showAlert(response.error || 'Error processing your request', 'error');
                }
            },
            error: function(xhr) {
                const errorMsg = xhr.responseJSON?.error || 'Server error occurred';
                showAlert(errorMsg, 'error');
                console.error('Prediction error:', xhr);
            }
        });
    });

    // Display loading state
    function showLoading(drug, food) {
        $('#output').html(`
            <div class="loading-state">
                <div class="spinner"></div>
                <p>Analyzing interaction between <strong>${drug}</strong> and <strong>${food}</strong>...</p>
            </div>
        `);
    }

    // Show alert message
    function showAlert(message, type) {
        $('#output').html(`
            <div class="alert alert-${type}">
                <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 'info-circle'}"></i>
                <span>${message}</span>
            </div>
        `);
    }

    // Display prediction results
    function displayPredictionResults(data) {
        const riskClass = data.risk_level.toLowerCase();
        const mechanismText = data.mechanism.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        
        let mechanismDetails = '';
        if (data.mechanism !== 'unknown') {
            mechanismDetails = `
                <div class="mechanism-details">
                    <h4><i class="fas fa-microscope"></i> Interaction Mechanism</h4>
                    <p>${data.explanation}</p>
                    <div class="mechanism-tag">${mechanismText}</div>
                </div>
            `;
        }
        
        let similarInteractions = '';
        if (data.similar_interactions && data.similar_interactions.length > 0) {
            similarInteractions = `
                <div class="similar-interactions">
                    <h4><i class="fas fa-random"></i> Similar Interactions</h4>
                    <div class="similar-list">
                        ${data.similar_interactions.map(interaction => `
                            <div class="similar-item">
                                <div class="similar-pair">${interaction.drug} + ${interaction.food}</div>
                                <div class="similar-risk ${interaction.risk_level.toLowerCase()}-risk">${interaction.risk_level}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }
        
        $('#output').html(`
            <div class="prediction-header">
                <div class="drug-food-pair">
                    <span class="drug-name">${data.drug}</span>
                    <span class="interaction-plus">+</span>
                    <span class="food-name">${data.food}</span>
                </div>
                <div class="prediction-confidence">
                    <span class="confidence-tag">${Math.round(data.probability * 100)}% Confidence</span>
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
                </div>
            </div>
        `);
        
        // Add event listeners for action buttons
        $('.btn-save-to-profile').click(function() {
            saveToProfile(data);
        });
        
        $('.btn-generate-report').click(function() {
            generateReport(data);
        });
    }

    // Save to profile function
    function saveToProfile(data) {
        // In a real implementation, this would make an API call to save to user's profile
        alert('This would save the interaction to your profile when logged in');
    }

    // Generate PDF report function
    function generateReport(data) {
        // In a real implementation, this would call your Flask endpoint to generate a PDF
        alert('This would generate a PDF report when implemented');
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
