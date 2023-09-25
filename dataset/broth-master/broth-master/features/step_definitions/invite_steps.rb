Given /^no invites exist$/ do
  Invite.delete_all
end

Given /^an invite exists for "([^\"]*)"$/ do |email|
  Invite.create(:email => email)
end
