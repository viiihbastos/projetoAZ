/**
 * Marketo Lead Collector for Notebooks
 * For more info and Documentation: https://databricks.atlassian.net/wiki/spaces/~863838343/pages/2080211260/How+to+use+Marketo+Lead+Collector+for+Notebooks
 */

function db_addStylesheetURL(url) {
    var link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = url;
    document.getElementsByTagName('head')[0].appendChild(link);
}
db_addStylesheetURL('https://fonts.googleapis.com/css?family=DM+Sans:400,500');

(function () {
    // Init form js
    var s = document.createElement('script');
    s.type = 'text/javascript';
    s.async = true;
    s.src = '//app-sj01.marketo.com/js/forms2/js/forms2.min.js';
    s.onreadystatechange = function () {
        if (this.readyState == 'complete' || this.readyState == 'loaded') {
            initMkto2();
        }
    };
    s.onload = initMkto2;
    document.getElementsByTagName('head')[0].appendChild(s);
    // Init formjs

    var get_attributes = document.getElementById("lead-collector");

    function initMkto2() {
        MktoForms2.loadForm("//pages.databricks.com", "094-YMS-629", 1001);

        // Create the form element
        var form = document.createElement("mktoForm_1001");
        form.setAttribute("method", "post");

        // Custom Marketo form lightbox behaviors
        // Remove old button
        document.querySelector(".tb-import a").remove();

        // Crete new button
        var notebook = document.getElementById('static-notebook');
        notebook.insertAdjacentHTML('beforeBegin', '' + '<style>.download_button{font-size:1rem;font-weight:600;color:#fff;min-width:154px;height:23px;padding:7px 9px;display:-webkit-box;display:flex;-webkit-box-align:center;-ms-flex-align:center;align-items:center;-webkit-box-pack:center;-ms-flex-pack:center;justify-content:center;background-color:#FD5000;border-color:#FD5000;margin:5px 10px;font-family:\'Barlow\',sans-serif;border-radius:.25rem;text-decoration:none;}.download_button:hover, .download_button:focus{background:#CA4000;color:#fff;text-decoration:none;}.download_button i{margin-right:5px;}' +
            'body .mktoButtonRow .mktoButtonWrap {margin-left: 0px !important;}' +
            'body .mktoButtonRow {margin-top: 10px !important;}' +
            'body .mktoButtonRow .mktoButton {     background-color: #FF3621 !important;\n' +
            '    font-weight: 500 !important;\n' +
            '    padding: 14px 38px !important;\n' +
            '    line-height: 25px !important;\n' +
            '    font-size: 18px !important;\n' +
            '    background-image: none !important;\n' +
            '    box-shadow: 0 !important;\n' +
            '    border: 0 !important; }' +
            '' +
            'body .mktoModal .mktoModalContent { padding: 40px; }' +
            'body .mktoModal .mktoForm * { font-family: "DM Sans", sans-serif !important; }' +
            'body .mktoModal #form_title { border-bottom: 0 !important; }' +
            'body .mktoModal .mktoModalClose { top: 15px; right: 15px; background: #ffff; font-size: 28px; line-height: 28px; transform: scaleY(0.8);\n' +
            '    border: 0px;\n' +
            '    color: #000; }' +
            'body .mktoModal .mktoLabel { font-size: 16px;\n' +
            '    font-weight: 500 !important; font-family: Helvetica, Arial, sans-serif !important;\n' +
            '    line-height: 20px;\n' +
            '    color: #1B5162;\n' +
            '}' +
            'body .mktoModal .mktoForm  input[type=\'text\'], body .mktoModal .mktoForm  input[type=\'email\'], body .mktoModal .mktoForm  input[type=\'tel\'], body .mktoModal .mktoForm  select, .mktoPopupForm input[type=\'text\'], .mktoPopupForm input[type=\'email\'], .mktoPopupForm input[type=\'tel\'], .mktoPopupForm select { width: 100% !important;\n' +
            '    font-weight: normal;\n' +
            '    font-size: 16px;\n' +
            '    color: #1B5162;\n' +
            '    line-height: 20px;\n' +
            '    border: 1px solid #DCE0E2;\n' +
            '    -webkit-box-sizing: border-box;\n' +
            '    box-sizing: border-box;\n' +
            '    background: #FFF;\n' +
            '    margin-bottom: 10px;\n' +
            '    margin-top: 4px;\n' +
            '    padding: 5px 15px;' +
            '    box-shadow: none;' +
            '    border-radius: 0;' +
            '    height: 32px;' +
            '}' +
            'body .mktoModal .mktoForm .mktoOffset { display: none !important; }' +
            'body .mktoModal .mktoForm .mktoFieldDescriptor { width: 100%; }' +
            'body .mktoModal .mktoForm .mktoFieldWrap { width: 100%; }' +
            'body .mktoModal .mktoForm .mktoLabel { width: 100% !important; }' +
            'body .mktoModal .mktoForm .mktoErrorMsg {\n' +
            '    color: #FF3621;     position: relative;\n' +
            '    right: 0 !important;\n' +
            '    bottom: 38px !important;\n' +
            '    font-size: 12px;\n' +
            '    background-color: inherit; background-image: none;\n' +
            '    border: 0; border-radius: 0; box-shadow: none;\n' +
            '    text-shadow: none;\n' +
            '}' +
            'body .mktoModal .mktoForm .mktoError {\n' +
            '    position: relative;\n' +
            '    right: 0 !important;\n' +
            '    z-index: -1 !important;\n' +
            '}' +
            'body .mktoErrorArrowWrap { display: none !important; }' +
            'body .mktoModal .mktoForm .mktoLogicalField { width: auto !important; }' +
            'body .mktoModal .mktoForm #Lblmkto_form_consent { padding-top: 0 !important; }' +
            'body .mktoModal { height: 100%; }' +
            'body .mktoModal .mktoModalContent { max-height: 80%; overflow-y: auto; top: 15px !important; }' +
            '' +
            '</style>');

        var form_div = document.querySelector('.tb-import');
        form_div.insertAdjacentHTML('afterbegin', '<a class="download_button" title="Import Notebook" href="#"><i class="fa fa-cloud-download"></i>  Download all notebooks</a>');

        // Create Custom Marketo form lightbox behaviors
        MktoForms2.whenReady(function (form) {
            // Debugging
            // form.vals({"FirstName":"Test","LastName":"Test","Company":"Databricks","Email":"test@test.com","mkto_form_consent":"No"});

            var formEl = form.getFormElem()[0],
                hasNotYouLink = formEl.querySelector(".mktoNotYou"),
                mktoKnownVisitor = !!hasNotYouLink;

            var consent_selector_field_wrap = formEl.querySelector('#Lblmkto_form_consent').closest('.mktoFieldWrap');
            consent_selector_field_wrap.setAttribute("style","display: flex; flex-direction: row-reverse;");
			var github_url = get_attributes.getAttribute('github-url');
            if (!mktoKnownVisitor) {
                var lbx = MktoForms2.lightbox(form, {
                    onSuccess: function (vals, tyURL) {
                        form.getFormElem().html("<style>p{margin-bottom: 15px !important;}</style><div>\n" +
                            "    <h3 id=\"form_title\" style=\"font-size:25px; margin-bottom: 15px; border-bottom:1px solid #ccc\">Thank you for your interest.</h3>\n" +
                            "    <p class='how-to-import'><span class='githuburl'>Access the notebooks on <a href='"+ github_url +"'  target='_blank' rel='noopener noreferrer' style='color:#FD5000'>GitHub here</a>.</span></p>\n" +
                            "    <p class='how-to-use'>Clone solution accelerator repository in Databricks using <a href='https://www.databricks.com/product/repos'  target='_blank' rel='noopener noreferrer' style='color:#FD5000'>Databricks Repos</a></span></p>\n" +
                            "    <p class='signup-text'>New to Databricks? <a class='signup-link' href='https://databricks.com/try-databricks?itm_data=notebooks-form-trial' target='_blank' rel='noopener noreferrer'  style='color:#FD5000'>Try it now.</a></p>\n" +
                            "</div>");
                        form.getFormElem().css("text-align", "center");
                        return false;
                    }
                });

                // Display Marketo Form after click download button
                var download_btn = document.querySelector('.download_button');
                download_btn.addEventListener("click", function () {
                    lbx.show();

                    // Adding Form Title
                    var form_title = document.getElementById("form_title");

                    if (form_title === null) {
                        var mform = document.querySelector('.mktoFormRow');
                        mform.insertAdjacentHTML('beforebegin', '<h3 id="form_title" style="font-size:25px; margin-bottom: 25px; border-bottom:1px solid #ccc">Download the notebooks now</h3>');
                    }
                    // /Adding Form Title

                });
            }
        });
    }
})();

/**
 * Databricks Javascript Debugger
 */

var is_DB_Debug = false;
var db_debug = function(){}

function db_marketopages_getCookie(name) {
    var cookie_match = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
    if(cookie_match) {
        return cookie_match.pop();
    }
    return '';
}

var cookie_db_debug = db_marketopages_getCookie('wp-db_debug');
if(cookie_db_debug != '') {
    is_DB_Debug = true;
}

if (is_DB_Debug) db_debug = console.log.bind(window.console)

/**
 * Databricks Set Cookie
 */
function db_set_cookie(name, value, days) {
    var expires = "";
    if (days) {
        var date = new Date();
        date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
        expires = "; expires=" + date.toUTCString();
    }
    document.cookie = name + "=" + (value || "") + expires + "; path=/";
}

/**
 * Databricks MaxMind handler
 */

var db_country_code = 'US';

var db_determine_country = (function() {
    var db_country = null;

    var callbackFunction = null;
    var handleMaxMindResponse = function(geoipResponse) {
        db_country = {country_code: geoipResponse.country.iso_code, country_name: geoipResponse.country.names.en}
        db_set_cookie('db_country', JSON.stringify(db_country), 30);
    };

    var onSuccess = function(geoipResponse) {
        db_debug('Maxmind Response:');
        db_debug(geoipResponse);
        handleMaxMindResponse(geoipResponse);
        callbackFunction(db_country);
    };

    var onError = function(error) {
        db_debug('Maxmind Error Encountered:');
        db_debug(error);
        callbackFunction(db_country);
    };

    return function(callbackParam) {
        var cookie_db_country =  db_marketopages_getCookie('db_country');
        if(cookie_db_country != '') {
            cookie_db_country = JSON.parse(cookie_db_country);
            db_debug('MaxMind Skipped: Cookie found (' + cookie_db_country.country_code + ')');
            db_country = cookie_db_country;
            db_country_code = cookie_db_country.country_code;
            callbackParam(db_country);
        }
        else {
            if (typeof geoip2 !== 'undefined') {
                db_debug('MaxMind Enabled: Determining country()');
                callbackFunction = callbackParam;
                geoip2.country(onSuccess, onError);
            } else {
                db_debug('MaxMind Disabled: Script was not loaded');
                callbackParam(db_country);
            }
        }
    };
}());

// Returns current page language code based on url
function SiteLang_code() {
    var languages = ['en', 'ca', 'cs', 'cx', 'cy', 'da', 'de', 'eu', 'ck', 'es', 'gn', 'fi', 'fr', 'gl', 'ht', 'hu', 'it', 'ko', 'nb', 'nn', 'fy', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sv', 'th', 'tr', 'ku', 'zh', 'fb', 'af', 'sq', 'hy', 'az', 'be', 'bn', 'bs', 'bg', 'hr', 'nl', 'eo', 'et', 'fo', 'ka', 'el', 'gu', 'hi', 'is', 'id', 'ga', 'jv', 'kn', 'kk', 'ky', 'la', 'lv', 'li', 'lt', 'mi', 'mk', 'mg', 'ms', 'mt', 'mr', 'mn', 'ne', 'pa', 'rm', 'sa', 'sr', 'so', 'sw', 'tl', 'ta', 'tt', 'te', 'ml', 'uk', 'uz', 'vi', 'xh', 'zu', 'km', 'tg', 'ar', 'he', 'ur', 'fa', 'sy', 'yi', 'qc', 'qu', 'ay', 'se', 'ps', 'tl', 'gx', 'my', 'qz', 'or', 'si', 'rw', 'ak', 'nd', 'sn', 'cb', 'ha', 'yo', 'ja', 'jp', 'lg', 'br', 'zz', 'tz', 'co', 'ig', 'as', 'am', 'lo', 'ny', 'wo', 'ff', 'sc', 'ln', 'tk', 'sz', 'bp', 'ns', 'tn', 'st', 'ts', 'ss', 'ks', 've', 'nr', 'ik', 'su', 'om', 'em', 'qr', 'iu', 'kr'];
    var findLangPath = window.location.pathname.split('/')[1];
    if (findLangPath.length === 2 && languages.includes(findLangPath)) {
        return findLangPath;
    } else {
        var hostname_lower = window.location.hostname.toLowerCase();
        if (hostname_lower.endsWith('.getsmartling.com')) {
            var smartling_locale = hostname_lower.substr(0,2);
            switch(smartling_locale) {
                case 'ja':
                    return 'jp';
                    break;
                case 'ko':
                    return 'kr';
                    break;
                case 'de':
                    return 'de';
                    break;
                case 'fr':
                    return 'fr';
                    break;
                case 'it':
                    return 'it';
                    break;
                default:
                    break;
            }
        }

        var lang_attr = jQuery('html').attr('lang');
        if (typeof lang_attr !== 'undefined' && lang_attr !== false) {
            lang_attr = lang_attr.toLowerCase();

            if (languages.includes(lang_attr)) {
                return lang_attr;
            }
        }

        return 'en';
    }
}


var script = document.createElement("script");
script.setAttribute("src", "//js.maxmind.com/js/apis/geoip2/v2.1/geoip2.js");
script.setAttribute("async", "false");
var head = document.head;
head.insertBefore(script, head.firstChild);

script = document.createElement("script");
script.onload = function () {
    /**
     * Databricks Marketo Checkbox Handler
     */

    var DBConsentStrings = {
        en: {
            'show_checkbox_text': 'Yes, I would like to receive marketing communications regarding Databricks services, events and open source products. I understand I can update my preferences at any time.',
            'no_checkbox_text': 'By submitting, I agree to the processing of my personal data by Databricks in accordance with our <a href="https://databricks.com/privacypolicy" target="_blank" id="">Privacy Policy</a>. I understand I can update my preferences at any time.',
            'partner_text': 'I agree that by signing up for this event, I am expressly consenting to provide my registration information, including name, email, etc., to Databricks and to the co-sponsor(s) of the event. Databricks and each co-sponsor may use my information in accordance with their privacy policy (see the event details for applicable privacy policies).'
        },
        de: {
            'show_checkbox_text': 'Ja, ich würde gerne Marketinginformationen über Databricks Dienstleistungen, Veranstaltungen und Open-Source-Produkte erhalten. Mir ist bewusst, dass ich meine Einstellungen jederzeit ändern kann.',
            'no_checkbox_text': 'Hiermit erkläre ich mich mit der Verarbeitung meiner persönlichen Daten durch Databricks in Übereinstimmung mit unseren <a href="https://databricks.com/de/privacypolicy" target="_blank" id="">Datenschutzrichtlinien</a> einverstanden. Ich bin mir bewusst, dass ich meine Einstellungen jederzeit ändern kann.',
            'partner_text': 'Mit der Registrierung für diese Veranstaltung erkläre ich mich ausdrücklich damit einverstanden, dass meine Registrierungsdaten, einschließlich Name, E-Mail usw., an Databricks und die Co-Sponsoren der Veranstaltung weitergegeben werden dürfen. Databricks und jeder Co-Sponsor können meine Daten in Übereinstimmung mit ihren Datenschutzrichtlinien verwenden (siehe die Details der Veranstaltung für die geltenden Datenschutzrichtlinien).'
        },
        fr: {
            'show_checkbox_text': 'Oui, je souhaite recevoir les communications marketing concernant les services, les événements et les produits open source de Databricks. Je suis conscient(e) que je peux modifier mes préférences à tout moment.',
            'no_checkbox_text': 'En validant, j\'accepte que mes données personnelles soient traitées par Databricks conformément à sa <a href="https://databricks.com/fr/privacypolicy" target="_blank" id="">politique de confidentialité</a>. Je suis conscient(e) que je peux modifier mes préférences à tout moment.',
            'partner_text': "En m'inscrivant à cet événement, je consens expressément à ce que les informations relatives à mon inscription, notamment mon nom, mon adresse électronique, etc. soient communiquées à Databricks et aux partenaires de l'événement. Databricks et chacun des partenaires peuvent utiliser mes informations conformément à leur politique de confidentialité (voir les détails de l'événement pour en savoir plus sur les politiques de confidentialité applicables)."
        },
        it: {
            'show_checkbox_text': 'Sì, desidero ricevere le comunicazioni commerciali relative ai servizi, agli eventi e ai prodotti open source di Databricks. Sono consapevole di poter aggiornare le mie preferenze in qualunque momento.',
            'no_checkbox_text': 'Inviando, fornisco il consenso al trattamento dei miei dati personali da parte di Databricks in conformità alla nostra <a href="https://databricks.com/it/privacypolicy" target="_blank" id="">Politica sulla Privacy</a>. Sono consapevole di poter aggiornare le mie preferenze in qualunque momento.',
            'partner_text': 'Iscrivendomi a questo evento, acconsento espressamente a fornire i miei dati di registrazione, compresi nome, indirizzo e-mail, ecc. a Databricks e agli altri sponsor dell\'evento. Databricks e ciascun altro sponsor possono usare i miei dati in conformità alla loro politica sulla privacy (vedi i dettagli dell\'evento per le politiche sulla privacy applicabili).'
        },
        kr: {
            'show_checkbox_text': '예, Databricks 서비스, 이벤트 및 오픈소스 제품과 관련한 마케팅 전달 사항을 받아보겠습니다. 언제든 기본 설정을 업데이트할 수 있다는 점을 이해했습니다.',
            'no_checkbox_text': '제출을 클릭해 Databricks가 본인의 개인정보 데이터를 자사 <a href="https://databricks.com/kr/privacypolicy" target="_blank" id="">개인정보 보호정책</a>에 따라 처리하는 데 동의합니다. 언제든 기본 설정을 업데이트할 수 있다는 점을 이해합니다.',
            'partner_text': '본인은 이 이벤트에 등록하여 Databricks 및 이벤트 공동 후원사에 본인의 등록 정보(이름, 이메일 등 포함)를 제공하기로 분명히 동의합니다. Databricks 및 각각의 공동 후원사에서는 본인의 정보를 각 회사의 개인정보 보호정책에 따라 사용할 수 있습니다(관련 개인정보 보호정책은 이벤트 상세 정보 참조).'
        },
        jp: {
            'show_checkbox_text': '私は、Databricks のサービス、イベント、オープンソース製品に関するマーケティング情報の受信を希望します。私は、この選択を随時変更できることを理解しています。',
            'no_checkbox_text': '送信することにより、Databricks が当社の<a href="https://databricks.com/jp/privacypolicy" target="_blank" id="">プライバシーポリシー</a>に定める範囲で私の個人データを利用することに同意します。私は、この選択を随時変更できることを理解しています。',
            'partner_text': '私は、このイベントに登録することにより、私の登録情報（氏名、メールアドレスなど）をDatabricks およびイベントの共催者に提供することに明示的に同意します。Databricks と各共催者は、それぞれのプライバシーポリシーに定める範囲で私の個人情報を利用できます。（※該当するプライバシーポリシーについては、イベントの詳細をご参照ください。）'
        }
    };

    var DBMarketoPartnerFormsArray = [1376,1588,1589,1794,2036,2068,2102,2247,2963,2973,3054,3120,3151,3195,3475,3536,3847,3876,3955,4271,4303,4364,4374,4481,4541,4549,4690,4748,4881,4924,4930,5076,5139,5429,5537,5541,5562];

    var country_selector_change_count = 0;

    function get_local_string(field, form_selector) {
        var lang_code = null;

        db_debug('DBMarketoConsentHandler: get_local_string(' + field + ')');

        // Check for form edge cases (e.g. JP form that is not using /jp URI pattern)
        if (form_selector != null) {
            var country_selector = form_selector.find('#Country');
            if (country_selector.length) {
                var us_option_selector = country_selector.find('option[value="United States"]');
                if (us_option_selector.length) {
                    switch(us_option_selector.html()) {
                        case 'Vereinigten Staaten von Amerika':
                            lang_code = 'de';
                            break;
                        case 'Étas-Unis':
                            lang_code = 'fr';
                            break;
                        case 'Stati Uniti':
                            lang_code = 'it';
                            break;
                        case '미합중국':
                            lang_code = 'kr';
                            break;
                        case 'アメリカ合衆国':
                            lang_code = 'jp';
                            break;
                        default:
                            break;
                    }

                    if (lang_code != null && DBConsentStrings.hasOwnProperty(lang_code) && DBConsentStrings[lang_code].hasOwnProperty(field)) {
                        return DBConsentStrings[lang_code][field];
                    }
                }
            }
        }

        lang_code = SiteLang_code();
        if (DBConsentStrings.hasOwnProperty(lang_code) && DBConsentStrings[lang_code].hasOwnProperty(field)) {
            return DBConsentStrings[lang_code][field];
        }

        return DBConsentStrings.en[field];
    }

    function marketoFormHandler(db_country, form, form_selector) {
        var country_code = null;
        var country_name = null;
        if (db_country != null) {
            country_code = db_country.country_code;
            country_name = db_country.country_name;
        }

        if (form_selector == null) {
            form_selector = jQuery('#mktoForm_' + form.getId());
        }
        var country_selector = form_selector.find('#Country');
        var consent_selector = form_selector.find('#mkto_form_consent');

        // Check for bypass class and disable script functionality if found
        var bypass_script = false;

        if (jQuery('.gtm-bypass-opt-in').length) {
            bypass_script = true;
        }

        if (country_selector.length) {
            db_debug('db_marketo_consent_handler: Country selector found');

            // Set country_selector value if found
            var was_country_found = false;
            country_selector.find('option').each(function () {
                if (this.value == country_name) {
                    db_debug('db_marketo_consent_handler: Found matching country');
                    was_country_found = true;
                    country_selector.val(country_name);
                    return false;
                }
            });

            if (!was_country_found) {
                db_debug('db_marketo_consent_handler: No country match found');
            }

            if (!bypass_script) {
                // Apply handler for country name changes
                country_selector.on('change', countrySelectorChangeHandler);

                // Trigger handler
                db_debug('db_marketo_consent_handler: Trigger change');
                country_selector.trigger('change');

                // Sanity check fallback timer
                setTimeout(function() {
                    db_debug('db_marketo_consent_handler: Fallback trigger country change');
                    country_selector.trigger('change');
                }, "500")
            }
        } else {
            db_debug('db_marketo_consent_handler: Country selector not found');

            if (!bypass_script) {
                // Show default if no Country selector
                if (consent_selector.length) {
                    showCheckboxVariant(form_selector, 'show_checkbox');
                }
            }
        }

        if (!bypass_script) {
            if (consent_selector.length) {
                db_debug('db_marketo_consent_handler: Consent checkbox found');

                // check if should show partner text
                // if so, create partner text DOM
                if(form != null && DBMarketoPartnerFormsArray.includes(form.getId())) {
                    db_debug('db_marketo_consent_handler: Activating partners text');
                    createPartnerTextBox(form_selector, consent_selector);
                } else {
                    db_debug('db_marketo_consent_handler: No partners text');
                }
            } else {
                db_debug('db_marketo_consent_handler: Consent checkbox not found');
            }
        }
    }

    function maxmindCallback(db_country) {
        var country_code = null;
        var country_name = null;
        if (db_country != null) {
            country_code = db_country.country_code;
            country_name = db_country.country_name;
        }
        db_debug('Determined country code: ' + country_code);
        db_debug('Determined country name: ' + country_name);
        db_debug('Marketo Form Library Detected: Applying onFormRender() logic');
        MktoForms2.whenRendered(function (form) {
            db_debug('db_marketo_consent_handler: MktoForms2.whenRendered(' + form.getId() + ')');
            marketoFormHandler(db_country, form, jQuery('#mktoForm_' + form.getId()));
        });

        // Marketo Trial Forms that are pre-rendered
        jQuery('.dbMarketoTrialForm').each(function() {
            var form_selector = jQuery(this);
            db_debug('db_marketo_consent_handler: dbMarketoTrialForm found');
            marketoFormHandler(db_country, null, form_selector);
        });
    };

    function countrySelectorChangeHandler() {
        // db_debug('db_marketo_consent_handler: countrySelectorChangeHandler: Country change detected (' + jQuery(this).val() + '): Change count(' + country_selector_change_count + ')');

        var this_selector = jQuery(this);
        var form_selector = this_selector.closest('.mktoForm');

        country_selector_change_count++;

        // determine what to show for checkbox label
        var checkbox_variant = getCheckboxVariant(this_selector.val());
        showCheckboxVariant(form_selector, checkbox_variant);

        // Sanity check fallback timer
        if (country_selector_change_count < 5) {
            var country_selector = form_selector.find('#Country, #country, input[name=Country]');
            setTimeout(function() {
                country_selector.trigger('change');
            }, "1000")
        }
    }

    function getCheckboxVariant(country_name) {
        var show_checkbox_country_names = ['', 'Germany', 'Austria', 'Switzerland', 'Canada', 'Australia', 'Sweden', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Singapore', 'United Kingdom'];

        if (show_checkbox_country_names.includes(country_name) || typeof country_name === 'undefined') {
            db_debug('db_marketo_consent_handler: getCheckboxVariant(' + country_name + '): show_checkbox');
            return 'show_checkbox';
        }

        db_debug('db_marketo_consent_handler: getCheckboxVariant(' + country_name + '): no_checkbox');
        return 'no_checkbox';
    }

    function showCheckboxVariant(form_selector, checkbox_variant) {
        var consent_selector = form_selector.find('#mkto_form_consent');

        if (consent_selector.length) {
            db_debug('db_marketo_consent_handler: Consent checkbox found');

            var consent_selector_row = consent_selector.closest('.mktoFormRow');
            var consent_selector_box = consent_selector_row.find('.mktoFieldDescriptor');
            var consent_selector_label = consent_selector_row.find('#Lblmkto_form_consent.mktoLabel');

            // check if variants have been created yet
            // if not, create them
            if (!form_selector.find('.dbMarketoConsentVariant').length) {
                // change existing to variant show_checkbox
                consent_selector_label.html(get_local_string('show_checkbox_text', form_selector));
                consent_selector_label.addClass('dbMarketoConsentVariantLabel');
                consent_selector_box.addClass('dbMarketoConsentVariant');
                consent_selector_box.addClass('mktoConsentFieldDescriptor');

                // special CSS class changes for campaign only form styling (mktocontact and mktpopup styles break if these rules are applied)
                if (form_selector.hasClass('mktcampaign')) {
                    consent_selector_row.find('.mktoLogicalField.mktoCheckboxList').removeClass('mktoLogicalField'); // remove weird styling behavior in main stylesheet
                    consent_selector_row.addClass('dbMarketoConsentRow'); // fix float CSS height behavior
                }

                // create no_checkbox variant
                var no_checkbox_variant_box = consent_selector_box.clone();
                no_checkbox_variant_box.addClass('dbMarketoConsentVariantNoCheckbox');
                no_checkbox_variant_box.find('input,label').remove();
                no_checkbox_variant_box.removeClass('mktoFieldDescriptor'); // this causes JS issues with Marketo library
                no_checkbox_variant_box.addClass('mktoConsentFieldDescriptor');
                no_checkbox_variant_box.find('.mktoFieldWrap').prepend(jQuery('<label class="mktoLabel mktoHasWidth dbMarketoConsentVariantLabel">' + get_local_string('no_checkbox_text', form_selector) + '</label>'));
                no_checkbox_variant_box.hide();
                no_checkbox_variant_box.insertAfter(consent_selector_box);

                // After the 'No' clone, can add 'Yes' specific classes now
                consent_selector_box.addClass('dbMarketoConsentVariantYesCheckbox');
            }

            switch(checkbox_variant) {
                case 'no_checkbox':
                    form_selector.find('.dbMarketoConsentVariantYesCheckbox').hide();
                    form_selector.find('.dbMarketoConsentVariantNoCheckbox').show();
                    consent_selector.prop( "checked", true );
                    form_selector.find('.dbMarketoOnlyShowYesCheckbox').hide();
                    break;
                case 'show_checkbox':
                default:
                    form_selector.find('.dbMarketoConsentVariantNoCheckbox').hide();
                    form_selector.find('.dbMarketoConsentVariantYesCheckbox').show();
                    consent_selector.prop( "checked", false );
                    form_selector.find('.dbMarketoOnlyShowYesCheckbox').show();
                    break;
            }
        }

    }

    function createPartnerTextBox(form_selector, consent_selector) {
        // Search for extra consent text instances that should be hidden
        form_selector.find('.mktoHtmlText').each(function() {
            var thisHTML = jQuery(this).html();
            if (thisHTML.length && thisHTML.includes('provide my registration information, including name, email')) {
                jQuery(this).closest('.mktoFormRow').hide();
            }
        });

        var consent_selector_row = consent_selector.closest('.mktoFormRow');
        jQuery('<div class="mktoFormRow"><div class="mktoFormCol dbPartnersMarketoText"><div class="mktoOffset mktoHasWidth"></div><div class="mktoFieldWrap"><div class="mktoHtmlText mktoHasWidth">' + get_local_string('partner_text', form_selector) + '</div><div class="mktoClear"></div></div><div class="mktoClear"></div></div><div class="mktoClear"></div></div>').insertBefore(consent_selector_row);
    }

    function DBMarketoConsentHandler_initialize() {
        if (typeof MktoForms2 !== 'undefined') {
            db_debug('Marketo Form Library Detected: Querying MaxMind');
            db_determine_country(maxmindCallback);
        } else {
            db_debug('Marketo Form Library Not Detected');
        }
    };

    // Initialize after document ready because DB theme loads Marketo in different prioritization
    jQuery(function() {
        var download_btn = document.querySelector('.download_button');
        download_btn.addEventListener("click", function () {
            DBMarketoConsentHandler_initialize();
        });
    });

};
script.setAttribute("src", "https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js?ver=5.8.3");
script.setAttribute("async", "false");
var head = document.head;
head.insertBefore(script, head.firstChild);


