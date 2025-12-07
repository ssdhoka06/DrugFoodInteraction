$(document).ready(function() {
    let API_BASE_URL = ''; // Use relative URLs for same domain
    
    // Check if API is running
    function checkAPIStatus() {
        console.log('🔍 Checking API connection...');
        $.ajax({
            url: '/api/drugs',
            type: 'GET',
            success: function(response) {
                console.log('✅ API is connected');
                console.log('API Response:', response);
                initializeSelect2();
            },
            error: function(xhr) {
                console.error('⚠️ API not responding:', xhr);
                console.error('Status:', xhr.status);
                console.error('Response:', xhr.responseText);
                showToast('Backend API is not responding. Please check if the server is running on port 5001.', 'error');
            }
        });
    }

    // Toast notification system
    function showToast(message, type = 'info') {
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };
        
        const colors = {
            success: 'bg-green-500',
            error: 'bg-red-500',
            warning: 'bg-yellow-500',
            info: 'bg-blue-500'
        };
        
        const toast = $(`
            <div class="fixed top-24 right-4 ${colors[type]} text-white px-6 py-4 rounded-2xl shadow-2xl flex items-center space-x-3 z-50 animate-slide-down">
                <i class="fas ${icons[type]} text-xl"></i>
                <span class="font-semibold">${message}</span>
            </div>
        `);
        
        $('body').append(toast);
        
        setTimeout(() => {
            toast.addClass('opacity-0 translate-x-full transition-all duration-500');
            setTimeout(() => toast.remove(), 500);
        }, 4000);
    }

    // Initialize Select2 dropdowns with enhanced styling
    function initializeSelect2() {
        $('#drug').select2({
            placeholder: "🔍 Type to search medications...",
            allowClear: true,
            width: '100%',
            minimumInputLength: 1,
            ajax: {
                url: '/api/drugs',
                dataType: 'json',
                delay: 250,
                data: function (params) {
                    return {
                        q: params.term,
                        limit: 20
                    };
                },
                processResults: function (data) {
                    return {
                        results: data.drugs.map(function(drug) {
                            return {
                                id: drug,
                                text: drug.charAt(0).toUpperCase() + drug.slice(1)
                            };
                        })
                    };
                },
                cache: true
            },
            templateResult: formatOption,
            templateSelection: formatSelection
        });

        $('#food').select2({
            placeholder: "🍽️ Type to search foods...",
            allowClear: true,
            width: '100%',
            minimumInputLength: 1,
            ajax: {
                url: '/api/foods',
                dataType: 'json',
                delay: 250,
                data: function (params) {
                    return {
                        q: params.term,
                        limit: 20
                    };
                },
                processResults: function (data) {
                    return {
                        results: data.foods.map(function(food) {
                            return {
                                id: food,
                                text: food.charAt(0).toUpperCase() + food.slice(1)
                            };
                        })
                    };
                },
                cache: true
            },
            templateResult: formatOption,
            templateSelection: formatSelection
        });

        loadInitialData();
    }

    // Format Select2 options with icons
    function formatOption(option) {
        if (!option.id) return option.text;
        return $('<span><i class="fas fa-pills text-primary-500 mr-2"></i>' + option.text + '</span>');
    }

    function formatSelection(option) {
        return option.text;
    }

    // Load initial data for dropdowns
    function loadInitialData() {
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

    // Handle form submission with enhanced animation
    $('#interactionForm').on('submit', function(e) {
        e.preventDefault();
        
        const drug = $('#drug').val();
        const food = $('#food').val();
        
        if (!drug || !food) {
            showToast('Please select both a medication and a food item', 'warning');
            return;
        }
        
        showLoading(drug, food);
        
        // Add timeout to detect hanging requests
        const timeout = setTimeout(function() {
            showAlert('Request is taking longer than expected. Please check if the server is running.', 'warning');
        }, 10000); // 10 second warning
        
        $.ajax({
            url: '/api/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                drug: drug,
                food: food
            }),
            timeout: 30000, // 30 second timeout
            success: function(response) {
                clearTimeout(timeout);
                console.log('API Response:', response);
                
                // Check if response has the expected structure
                if (response.error) {
                    showAlert(response.error, 'error');
                    showToast(response.error, 'error');
                    return;
                }
                
                displayPredictionResults(response);
                showToast('Analysis complete!', 'success');
                
                // Smooth scroll to results
                $('html, body').animate({
                    scrollTop: $('#output').offset().top - 100
                }, 800, 'swing');
            },
            error: function(xhr, status, error) {
                clearTimeout(timeout);
                console.error('AJAX Error:', {
                    status: status,
                    error: error,
                    response: xhr.responseText,
                    statusCode: xhr.status
                });
                
                let errorMsg = 'Server error occurred';
                
                if (xhr.status === 0) {
                    errorMsg = 'Cannot connect to server. Please ensure Flask is running on port 5001.';
                } else if (xhr.status === 404) {
                    errorMsg = 'API endpoint not found. Check if /api/predict exists.';
                } else if (xhr.status === 500) {
                    errorMsg = 'Server internal error. Check Flask console for details.';
                } else if (status === 'timeout') {
                    errorMsg = 'Request timed out. The server might be overloaded.';
                } else {
                    errorMsg = xhr.responseJSON?.error || error || 'Unknown error occurred';
                }
                
                showAlert(errorMsg, 'error');
                showToast(errorMsg, 'error');
            }
        });
    });

    // Enhanced loading state with animation
    function showLoading(drug, food) {
        $('#output').html(`
            <div class="flex flex-col items-center justify-center h-full py-12 animate-fade-in">
                <div class="relative w-24 h-24 mb-6">
                    <div class="absolute inset-0 border-4 border-primary-200 rounded-full"></div>
                    <div class="absolute inset-0 border-4 border-primary-600 rounded-full border-t-transparent animate-spin"></div>
                    <div class="absolute inset-2 bg-primary-100 rounded-full flex items-center justify-center">
                        <i class="fas fa-atom text-primary-600 text-2xl animate-pulse"></i>
                    </div>
                </div>
                <h3 class="text-xl font-bold text-gray-800 mb-2">Analyzing Interaction...</h3>
                <p class="text-gray-600 text-center max-w-md">
                    Processing <strong class="text-primary-600">${drug}</strong> and 
                    <strong class="text-secondary-600">${food}</strong> through our AI model
                </p>
                <div class="mt-6 flex space-x-2">
                    <div class="w-3 h-3 bg-primary-500 rounded-full animate-bounce" style="animation-delay: 0s;"></div>
                    <div class="w-3 h-3 bg-primary-500 rounded-full animate-bounce" style="animation-delay: 0.2s;"></div>
                    <div class="w-3 h-3 bg-primary-500 rounded-full animate-bounce" style="animation-delay: 0.4s;"></div>
                </div>
            </div>
        `);
    }

    // Enhanced alert display
    function showAlert(message, type) {
        const alertColors = {
            error: 'bg-red-50 border-red-200 text-red-800',
            warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
            info: 'bg-blue-50 border-blue-200 text-blue-800',
            success: 'bg-green-50 border-green-200 text-green-800'
        };
        
        const alertIcons = {
            error: 'fa-exclamation-circle text-red-600',
            warning: 'fa-exclamation-triangle text-yellow-600',
            info: 'fa-info-circle text-blue-600',
            success: 'fa-check-circle text-green-600'
        };
        
        $('#output').html(`
            <div class="animate-scale-in">
                <div class="border-2 rounded-2xl p-6 ${alertColors[type]} flex items-start space-x-4">
                    <i class="fas ${alertIcons[type]} text-2xl mt-1"></i>
                    <div class="flex-1">
                        <h4 class="font-bold mb-1">${type.charAt(0).toUpperCase() + type.slice(1)}</h4>
                        <p>${message}</p>
                    </div>
                </div>
            </div>
        `);
    }

    // Enhanced prediction results display
    function displayPredictionResults(data) {
        console.log('📊 Displaying results:', data);
        
        // Handle both direct response and wrapped response
        const result = data.data || data;
        
        const riskClass = result.risk_level.toLowerCase();
        const mechanismText = result.mechanism.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        
        const riskColors = {
            high: {
                gradient: 'from-red-500 to-pink-500',
                bg: 'bg-red-50',
                border: 'border-red-200',
                text: 'text-red-800',
                icon: 'fa-exclamation-triangle',
                badge: 'bg-red-100 text-red-800'
            },
            moderate: {
                gradient: 'from-yellow-500 to-orange-500',
                bg: 'bg-yellow-50',
                border: 'border-yellow-200',
                text: 'text-yellow-800',
                icon: 'fa-exclamation-circle',
                badge: 'bg-yellow-100 text-yellow-800'
            },
            low: {
                gradient: 'from-green-500 to-teal-500',
                bg: 'bg-green-50',
                border: 'border-green-200',
                text: 'text-green-800',
                icon: 'fa-check-circle',
                badge: 'bg-green-100 text-green-800'
            }
        };
        
        const risk = riskColors[riskClass] || riskColors.low;
        
        let mechanismDetails = '';
        if (result.mechanism !== 'unknown') {
            const mechanismExplanations = {
                'cyp3a4_inhibition': 'This food blocks liver enzymes that metabolize the medication, potentially causing dangerous buildup.',
                'vitamin_k_competition': 'This food contains vitamin K which can interfere with blood-thinning effects.',
                'calcium_chelation': 'Calcium in this food binds to the medication, reducing absorption.',
                'absorption_interference': 'This food can slow down or reduce medication absorption.'
            };
            
            const explanation = mechanismExplanations[result.mechanism] || 'The interaction mechanism requires further study.';
            
            mechanismDetails = `
                <div class="bg-white rounded-2xl p-6 border-2 ${risk.border} mt-6 animate-slide-up" style="animation-delay: 0.2s;">
                    <div class="flex items-center space-x-3 mb-4">
                        <div class="w-12 h-12 ${risk.bg} rounded-xl flex items-center justify-center">
                            <i class="fas fa-microscope ${risk.text} text-xl"></i>
                        </div>
                        <h4 class="text-lg font-bold text-gray-900">Interaction Mechanism</h4>
                    </div>
                    <p class="text-gray-700 mb-3">${explanation}</p>
                    <span class="inline-block px-4 py-2 ${risk.badge} rounded-xl text-sm font-semibold">${mechanismText}</span>
                </div>
            `;
        }
        
        const recommendations = {
            high: '🚨 <strong>Avoid this combination.</strong> Consult your healthcare provider immediately before combining these.',
            moderate: '⚠️ <strong>Use with caution.</strong> Space timing by 2-4 hours and monitor for side effects.',
            low: '✅ <strong>Generally safe.</strong> This combination poses minimal risk with normal precautions.'
        };
        
        $('#output').html(`
            <div class="animate-fade-in">
                <!-- Header Card -->
                <div class="bg-gradient-to-r ${risk.gradient} rounded-2xl p-6 text-white mb-6 shadow-xl">
                    <div class="flex items-center justify-between mb-4">
                        <div class="flex items-center space-x-4">
                            <div class="w-16 h-16 bg-white/20 backdrop-blur-sm rounded-2xl flex items-center justify-center">
                                <i class="fas ${risk.icon} text-3xl"></i>
                            </div>
                            <div>
                                <h3 class="text-2xl font-bold mb-1">
                                    ${result.interaction_predicted ? 'Interaction Detected' : 'No Significant Interaction'}
                                </h3>
                                <p class="text-white/90 font-semibold">${result.risk_level} Risk Level</p>
                            </div>
                        </div>
                        <div class="text-right">
                            <div class="text-3xl font-bold mb-1">${Math.round(result.probability * 100)}%</div>
                            <p class="text-sm text-white/90">Confidence</p>
                        </div>
                    </div>
                    <div class="flex items-center justify-center space-x-3 text-lg">
                        <span class="px-6 py-3 bg-white/20 backdrop-blur-sm rounded-xl font-bold">${result.drug}</span>
                        <i class="fas fa-plus"></i>
                        <span class="px-6 py-3 bg-white/20 backdrop-blur-sm rounded-xl font-bold">${result.food}</span>
                    </div>
                </div>
                
                ${mechanismDetails}
                
                <!-- Recommendations Card -->
                <div class="bg-white rounded-2xl p-6 border-2 ${risk.border} mt-6 animate-slide-up" style="animation-delay: 0.3s;">
                    <div class="flex items-center space-x-3 mb-4">
                        <div class="w-12 h-12 ${risk.bg} rounded-xl flex items-center justify-center">
                            <i class="fas fa-stethoscope ${risk.text} text-xl"></i>
                        </div>
                        <h4 class="text-lg font-bold text-gray-900">Clinical Recommendations</h4>
                    </div>
                    <p class="text-gray-700 leading-relaxed">${recommendations[riskClass]}</p>
                </div>
                
                <!-- Categories Card -->
                <div class="grid md:grid-cols-2 gap-4 mt-6">
                    <div class="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-2xl p-6 border-2 border-blue-100 animate-slide-up" style="animation-delay: 0.4s;">
                        <div class="flex items-center space-x-3 mb-3">
                            <i class="fas fa-pills text-blue-600 text-xl"></i>
                            <h5 class="font-bold text-gray-900">Drug Category</h5>
                        </div>
                        <p class="text-blue-900 font-semibold text-lg capitalize">${result.drug_category.replace(/_/g, ' ')}</p>
                    </div>
                    <div class="bg-gradient-to-br from-purple-50 to-pink-50 rounded-2xl p-6 border-2 border-purple-100 animate-slide-up" style="animation-delay: 0.5s;">
                        <div class="flex items-center space-x-3 mb-3">
                            <i class="fas fa-utensils text-purple-600 text-xl"></i>
                            <h5 class="font-bold text-gray-900">Food Category</h5>
                        </div>
                        <p class="text-purple-900 font-semibold text-lg capitalize">${result.food_category.replace(/_/g, ' ')}</p>
                    </div>
                </div>
                
                <!-- Action Buttons -->
                <div class="flex flex-col sm:flex-row gap-4 mt-6 animate-slide-up" style="animation-delay: 0.6s;">
                    <button class="btn-save-to-profile flex-1 px-6 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 text-white font-bold rounded-2xl hover:shadow-xl hover:scale-105 transition-all duration-300 flex items-center justify-center space-x-2">
                        <i class="fas fa-save"></i>
                        <span>Save to Profile</span>
                    </button>
                    <button class="btn-generate-report flex-1 px-6 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-bold rounded-2xl hover:shadow-xl hover:scale-105 transition-all duration-300 flex items-center justify-center space-x-2">
                        <i class="fas fa-file-pdf"></i>
                        <span>Generate PDF Report</span>
                    </button>
                </div>
            </div>
        `);
        
        // Add event listeners for action buttons with animations
        $('.btn-save-to-profile').click(function() {
            $(this).html('<i class="fas fa-check mr-2"></i>Saved!').addClass('from-green-600 to-teal-600');
            showToast('Interaction saved to your profile', 'success');
            setTimeout(() => {
                $(this).html('<i class="fas fa-save mr-2"></i>Save to Profile').removeClass('from-green-600 to-teal-600');
            }, 2000);
        });
        
        $('.btn-generate-report').click(function() {
            const originalHtml = $(this).html();
            $(this).html('<i class="fas fa-spinner fa-spin mr-2"></i>Generating...').prop('disabled', true);
            setTimeout(() => {
                $(this).html('<i class="fas fa-check mr-2"></i>Downloaded!').removeClass('from-purple-600 to-pink-600').addClass('from-green-600 to-teal-600');
                showToast('Report generated successfully', 'success');
                setTimeout(() => {
                    $(this).html(originalHtml).prop('disabled', false).removeClass('from-green-600 to-teal-600').addClass('from-purple-600 to-pink-600');
                }, 2000);
            }, 1500);
        });
    }

    // Clear form with animation
    $('.btn-clear').click(function() {
        $('#drug').val(null).trigger('change');
        $('#food').val(null).trigger('change');
        
        $('#output').html(`
            <div class="results-placeholder flex flex-col items-center justify-center h-full text-center animate-fade-in">
                <div class="w-24 h-24 bg-primary-100 rounded-3xl flex items-center justify-center mb-6 animate-pulse-slow">
                    <i class="fas fa-microscope text-primary-600 text-4xl"></i>
                </div>
                <h3 class="text-xl font-semibold text-gray-700 mb-2">Ready to Analyze</h3>
                <p class="text-gray-500 max-w-md">
                    Enter a drug and food combination above to see detailed interaction analysis powered by AI
                </p>
            </div>
        `);
        
        showToast('Form cleared', 'info');
    });

    // Modal handling with enhanced animations
    $('.btn-login, .btn-signup').click(function() {
        const modalId = $(this).hasClass('btn-login') ? '#loginModal' : '#signupModal';
        $(modalId).removeClass('hidden').addClass('flex');
        setTimeout(() => {
            $(modalId).find('.bg-white').removeClass('scale-90 opacity-0').addClass('scale-100 opacity-100');
        }, 10);
    });
    
    $('.modal-close').click(function() {
        const modal = $(this).closest('.modal');
        modal.find('.bg-white').removeClass('scale-100 opacity-100').addClass('scale-90 opacity-0');
        setTimeout(() => {
            modal.removeClass('flex').addClass('hidden');
        }, 300);
    });
    
    $('.switch-to-signup').click(function(e) {
        e.preventDefault();
        $('#loginModal').find('.modal-close').click();
        setTimeout(() => $('.btn-signup').click(), 300);
    });
    
    $('.switch-to-login').click(function(e) {
        e.preventDefault();
        $('#signupModal').find('.modal-close').click();
        setTimeout(() => $('.btn-login').click(), 300);
    });
    
    // Close modal when clicking outside
    $('.modal').click(function(e) {
        if ($(e.target).hasClass('modal')) {
            $(this).find('.modal-close').click();
        }
    });
    
    // Mobile menu toggle with animation
    $('.mobile-menu-btn').click(function() {
        const navLinks = $('.nav-links');
        navLinks.toggleClass('active');
        $(this).find('i').toggleClass('fa-bars fa-times');
    });
    
    // Smooth scrolling for navigation links
    $('a[href^="#"]').on('click', function(e) {
        e.preventDefault();
        
        $('.nav-links').removeClass('active');
        $('.mobile-menu-btn i').removeClass('fa-times').addClass('fa-bars');
        
        const target = $(this).attr('href');
        if (target === '#') return;
        
        $('html, body').animate({
            scrollTop: $(target).offset().top - 100
        }, 800, 'swing');
    });
    
    // Active nav link highlighting on scroll
    $(window).scroll(function() {
        const scrollPosition = $(this).scrollTop();
        
        $('.nav-link').removeClass('text-primary-600');
        
        $('section').each(function() {
            const currentSection = $(this);
            const sectionTop = currentSection.offset().top - 150;
            const sectionId = currentSection.attr('id');
            
            if (scrollPosition >= sectionTop && 
                scrollPosition < sectionTop + currentSection.outerHeight()) {
                $(`.nav-link[href="#${sectionId}"]`).addClass('text-primary-600');
            }
        });
        
        // Add shadow to navbar on scroll
        if (scrollPosition > 50) {
            $('nav').addClass('shadow-lg');
        } else {
            $('nav').removeClass('shadow-lg');
        }
    });
    
    // Parallax effect for hero section
    $(window).scroll(function() {
        const scrolled = $(this).scrollTop();
        $('.hero').css('transform', 'translateY(' + (scrolled * 0.5) + 'px)');
    });
    
    // Add hover effects to cards
    $('.dashboard-card, .research-card').hover(
        function() {
            $(this).addClass('transform -translate-y-2 shadow-2xl');
        },
        function() {
            $(this).removeClass('transform -translate-y-2 shadow-2xl');
        }
    );
    
    // Initialize with first section active
    $('.nav-link[href="#predictor"]').addClass('text-primary-600');
    
    // Loading animation on page load
    $(window).on('load', function() {
        $('body').addClass('loaded');
    });
    
    // Add ripple effect to buttons
    $('button, .btn-primary, .btn-secondary').on('click', function(e) {
        const ripple = $('<span class="ripple"></span>');
        $(this).append(ripple);
        
        const x = e.pageX - $(this).offset().left;
        const y = e.pageY - $(this).offset().top;
        
        ripple.css({
            left: x + 'px',
            top: y + 'px'
        });
        
        setTimeout(() => ripple.remove(), 600);
    });
    
    console.log('🚀 Drug-Food Interaction Predictor initialized successfully!');
});
