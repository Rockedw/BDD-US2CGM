Given /^site settings have been enabled for "(.*)"$/ do |setting|
  SiteSetting.first.update_attributes(setting => true)
end

Given /^site settings have been disabled for "(.*)"$/ do |setting|
  SiteSetting.first.update_attributes(setting => false)
end

Given /^an email template for "(.*)" exists$/ do |name|
  Factory.create(:email_template, :name => name)
end
