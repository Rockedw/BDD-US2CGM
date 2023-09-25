Then /^I should see error messages$/ do
  page.body.should match /error(s)? prohibited/m
end

Given /^no user exists with an email of "(.*)"$/ do |email|
  User.find_by_email(email).should be_nil
end

Given /^a user exists with an email of "(.*)"$/ do |email|
  user = Factory :valid_user, :login => "valid_user", :email => email
end

Given /^I am signed up and confirmed as "(.*)\/(.*)"$/ do |email, password|
  user = Factory :valid_user,
    :email                 => email,
    :password              => password,
    :password_confirmation => password
end

Given /^I am signed up and signed in as "(.*)\/(.*)"$/ do |email, password|
  user = Factory :valid_user,
    :email                 => email,
    :password              => password,
    :password_confirmation => password
  When %{I sign in as "#{email}\/#{password}"}
end

Given /^I am signed out$/ do
  When %{I go to the sign out page}
end

When /^I sign in( with "remember me")? as "(.*)\/(.*)"$/ do |remember, email, password|
  When %{I go to the sign in page}
  And %{I fill in "Email" with "#{email}"}
  And %{I fill in "Password" with "#{password}"}
  And %{I check "Remember me"} if remember
  And %{I press "Sign In"}
end

Then /^I should be signed in as "(.*)"$/ do |email|
  page.should have_content(email)
end

Then /^I should be signed out$/ do
  page.should have_content("Sign In")
end

