const translations = {
  en: {
    time: "10:00 am - 5:00 pm",
    "opening hours": "Opening Hour: Sunday - Friday",
    number: "+977 12346789",
    "number description": "Call Us For Free Consultation",
    home: "Home",
    about: "About",
    practice: "Practice",
    attorneys: "Attorneys",
    "case studies": "Case Studies",
    contact: "Contact",
    location: "Location",
    justice: "We fight for your justice"
  },
  
  ne: {
    kanun: "कानुन",
    time: "८:०० - ९:०० am",
    "opening hours": "खुल्ने समय: सोमबार - शुक्रबार",
    number: "+९७७ ०१२३४५६७८९",
    "number description": "निःशुल्क परामर्शको लागि हामीलाई सम्पर्क गर्नुहोस्",
    home: "गृहपृष्ठ",
    about: "हाम्रो बारेमा",
    practice: "अभ्यास",
    attorneys: "वकिलहरू",
    "case studies": "मामला अध्ययनहरू",
    contact: "सम्पर्क",
    location: "स्थान",
    justice: "हामी तपाईंको न्यायको लागि लड्छौं"
  }

};

function setLanguage(lang) {
  document.querySelectorAll("[data-key]").forEach(element => {
    const key = element.getAttribute("data-key");
    element.innerText = translations[lang][key];
  });
}

// default language
setLanguage("en");
